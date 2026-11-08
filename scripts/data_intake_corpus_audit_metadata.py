"""Legacy envelope support accounting; mirrors the existing public corpus reader."""
def adapter_support(row):
    if 'model_adapter_supported' in row:
        value = row['model_adapter_supported']
        if type(value) is not bool:
            raise ValueError('explicit adapter support must be boolean')
        return value
    return row['native_schema'] == 'mora3.ObservationalSciPlexTask/v1'
