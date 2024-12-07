import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;

public class WineQualityTrainer {
    private static final String awsRegion = "us-east-1"; // Add missing variable

    public static void main(String[] args) {
        // Initialize Spark session with t2.micro cluster configurations
        SparkSession spark = SparkSession.builder()
            .appName("Wine Quality Trainer")
            .config("spark.master", "spark://ip-172-31-90-41.ec2.internal:7077")  // Replace MASTER_IP with your master node IP
            // Updated AWS configurations
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                   "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider")
            .config("spark.hadoop.fs.s3a.access.key", System.getenv("AWS_ACCESS_KEY_ID"))
            .config("spark.hadoop.fs.s3a.secret.key", System.getenv("AWS_SECRET_ACCESS_KEY"))
            .config("spark.hadoop.fs.s3a.session.token", System.getenv("AWS_SESSION_TOKEN"))
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.endpoint", "s3.us-east-1.amazonaws.com")
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            // t2.micro cluster optimized configurations
            .config("spark.driver.memory", "750m")
            .config("spark.executor.memory", "750m")
            .config("spark.executor.cores", "1")
            .config("spark.executor.instances", "3")  // One executor per worker
            .config("spark.default.parallelism", "3")
            .config("spark.sql.shuffle.partitions", "3")
            .config("spark.memory.fraction", "0.8")
            .config("spark.driver.maxResultSize", "500m")
            .config("spark.rdd.compress", "true")
            .config("spark.shuffle.compress", "true")
            .getOrCreate();

        // Define schema to properly handle the CSV format
        StructType schema = new StructType(new StructField[] {
            new StructField("fixed_acidity", DataTypes.DoubleType, false, Metadata.empty()),
            new StructField("volatile_acidity", DataTypes.DoubleType, false, Metadata.empty()),
            new StructField("citric_acid", DataTypes.DoubleType, false, Metadata.empty()),
            new StructField("residual_sugar", DataTypes.DoubleType, false, Metadata.empty()),
            new StructField("chlorides", DataTypes.DoubleType, false, Metadata.empty()),
            new StructField("free_sulfur_dioxide", DataTypes.DoubleType, false, Metadata.empty()),
            new StructField("total_sulfur_dioxide", DataTypes.DoubleType, false, Metadata.empty()),
            new StructField("density", DataTypes.DoubleType, false, Metadata.empty()),
            new StructField("pH", DataTypes.DoubleType, false, Metadata.empty()),
            new StructField("sulphates", DataTypes.DoubleType, false, Metadata.empty()),
            new StructField("alcohol", DataTypes.DoubleType, false, Metadata.empty()),
            new StructField("quality", DataTypes.DoubleType, false, Metadata.empty())
        });

        // Load training data with reduced partitions
        Dataset<Row> trainingData = spark.read()
            .option("header", "true")
            .option("delimiter", ";")
            .option("quote", "\"")
            .schema(schema)
            .csv("s3a://cs643-wine-quality-datasets/TrainingDataset.csv")
            .repartition(3);  // One partition per worker

        // Load validation data with reduced partitions
        Dataset<Row> validationData = spark.read()
            .option("header", "true")
            .option("delimiter", ";")
            .option("quote", "\"")
            .schema(schema)
            .csv("s3a://cs643-wine-quality-datasets/ValidationDataset.csv")
            .repartition(3);  // One partition per worker

        // Combine all feature columns into a vector column
        String[] featureColumns = {
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
            "pH", "sulphates", "alcohol"
        };

        // Create vector assembler with parallel processing
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(featureColumns)
            .setOutputCol("features")
            .setHandleInvalid("skip");

        // Initialize RandomForest classifier with reduced complexity
        RandomForestClassifier rf = new RandomForestClassifier()
                    .setLabelCol("quality")
            .setFeaturesCol("features")
            .setNumTrees(30)          // Reduced for t2.micro
            .setMaxDepth(5)           // Reduced for t2.micro
            .setMaxBins(16)           // Reduced for t2.micro
            .setSeed(42)
            .setSubsamplingRate(0.6)  // Reduced for t2.micro
            .setCacheNodeIds(false);   // Save memory

        // Create pipeline
        Pipeline pipeline = new Pipeline()
            .setStages(new PipelineStage[] {assembler, rf});

        // Train model
        System.out.println("Training model...");
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions on validation data
        Dataset<Row> predictions = model.transform(validationData);

        // Calculate F1 Score
        MulticlassMetrics metrics = new MulticlassMetrics(predictions.select("prediction", "quality"));
        double f1Score = metrics.weightedFMeasure();
        System.out.println("Weighted F1 Score on validation data: " + f1Score);

        // Save model to S3 with error handling
        try {
            System.out.println("Saving model to S3...");
            String modelPath = String.format("s3a://cs643-wine-quality-datasets/%s/wine-quality-model", awsRegion);
            model.write().overwrite().save(modelPath);
            System.out.println("Model saved successfully to: " + modelPath);
        } catch (Exception e) {
            System.err.println("Error saving model: " + e.getMessage());
            e.printStackTrace();
        } finally {
            spark.stop();
        }
    }
}
