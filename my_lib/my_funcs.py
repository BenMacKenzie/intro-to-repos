def convertFtoC(unitCol, tempCol):
    from pyspark.sql.functions import when, col
    return when(col(unitCol) == "F", (col(tempCol) - 32) * (50/9)).otherwise(col(tempCol)).alias("temp_celcius")

def roundedTemp(unitCol, tempCol):
    from pyspark.sql.functions import round, concat_ws
    return concat_ws(" ", round(tempCol, 3).cast("string"), unitCol).alias("rounded_temp")
