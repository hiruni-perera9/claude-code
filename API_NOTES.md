# PaleoDB API Notes

## Issue: API 400 Error

The PaleoDB API can sometimes return a 400 error due to:
- Invalid API parameters
- API changes or downtime
- Rate limiting
- Network restrictions in deployment environments

## Solution: Automatic Fallback

The data loader now includes automatic fallback to synthetic data:

```python
loader = PaleoDBLoader()
data = loader.download_paleodb_data(limit=10000)
# Will use real API if available, or generate synthetic data
```

### How It Works

1. **First attempt**: Try to fetch from PaleoDB API with corrected parameters
2. **On failure**: Automatically generate synthetic fossil occurrence data
3. **Seamless training**: Model trains successfully either way

### Synthetic Data

When the API is unavailable, the system generates realistic synthetic data with:
- Taxonomic information (phylum, class, order)
- Geographic coordinates (longitude, latitude)
- Paleographic coordinates
- Temporal data (geological ages)
- Environmental information
- Collection metadata

The synthetic data has similar statistical properties to real PaleoDB data and is suitable for:
- ✅ Training and testing the anomaly detection model
- ✅ Demonstrating the dashboard
- ✅ Development and deployment
- ⚠️ Not suitable for real scientific research

### Using Real PaleoDB Data

For production use with real data:

**Option 1: Fix API connectivity**
- Ensure network access to paleobiodb.org
- Check firewall/proxy settings
- Verify API endpoint is accessible

**Option 2: Use cached data**
- Download data once and cache it
- Store in `./data/` directory
- Load from cache instead of API

**Option 3: Upload your own data**
- Use the dashboard's "Upload CSV" feature
- Provide PaleoDB-formatted CSV files
- Must have similar structure to API data

## API Parameters

Corrected API parameters for PaleoDB v1.2:

```python
params = {
    "limit": 10000,
    "show": "coords,loc,paleoloc,class,classext",
    "vocab": "pbdb",
}
```

Common invalid parameters that cause 400 errors:
- ❌ `show: "class"` alone (too ambiguous)
- ❌ Missing `vocab` parameter
- ❌ Invalid `show` options

## Troubleshooting

### Still getting API errors?

1. **Check API status**: Visit https://paleobiodb.org
2. **Test API directly**:
   ```bash
   curl "https://paleobiodb.org/data1.2/occs/list.json?limit=10&show=coords,loc&vocab=pbdb"
   ```
3. **Use synthetic data**: The fallback will work automatically

### Need real data urgently?

Download a CSV export from PaleoDB website:
1. Visit https://paleobiodb.org/navigator/
2. Configure your query
3. Download as CSV
4. Upload to dashboard

## For Deployment

The automatic fallback ensures:
- ✅ Training always succeeds
- ✅ Dashboard always works
- ✅ No manual intervention needed
- ✅ Works on Streamlit Cloud, Render, Railway

The system will display whether it's using real or synthetic data.
