"""Exact typed-value roundtrip independent of Parquet's list item/element field name."""
def native_table_equal(original,decoded):
    # Top-level column names, data types and every value are still compared.
    # No native source metadata is attached to these typed reference tables.
    return original.equals(decoded,check_metadata=False)
