from pyspark.sql.types import IntegerType, StringType, StructField, StructType

MTSAMPLES_SCHEMA = StructType(
    [
        StructField("id", IntegerType(), nullable=False),
        StructField("description", StringType(), nullable=True),
        StructField("medical_specialty", StringType(), nullable=True),
        StructField("sample_name", StringType(), nullable=True),
        StructField("transcription", StringType(), nullable=True),
        StructField("keywords", StringType(), nullable=True),
    ]
)
