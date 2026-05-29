# Bird Migration Path Explorer

Interactive browser app for visualizing bird migration paths from:

`data/filtered/dataset2_daily_movement_lat30_50_birds100_all_month_context.csv`

## Run the App

From the project root:

```powershell
python -m http.server 8765 --bind 127.0.0.1
```

Open this URL in a browser:

```text
http://127.0.0.1:8765/migration_app/index.html
```

The app should be served from the project root so it can load the CSV using the relative path.

## How to Use

1. Select a bird from the **Bird** dropdown.
2. View the selected bird's migration path on the GPS latitude/longitude grid.
3. Follow the color ramp:
   - Blue points are earlier records.
   - Green/yellow points are middle records.
   - Red points are later records.
4. Hover over any point to inspect:
   - Date
   - Latitude and longitude
   - Daily step distance
   - Speed
   - GPS point count
   - Stopover duration
5. Adjust **Flyday threshold** to change how flydays are counted. The default is `30 km/day`.
6. Use **Fit Bird** to zoom the grid to the selected bird's path.
7. Use **Fit All** to keep the grid scaled to the full dataset extent.

## Summary Metrics

For the selected bird, the sidebar shows:

- **Start**: first recorded date.
- **End**: last recorded date.
- **Distance**: total traveled distance, computed from `step_length_km`.
- **Flydays**: number of days where `step_length_km >= threshold`.
- **Days tracked**: calendar days between the first and last record.
- **Records**: number of daily records for that bird.
- **Latitude range** and **Longitude range**: spatial extent of the selected bird's data.
- **Longest daily move**: maximum `step_length_km`.

## Files

- `index.html`: app layout.
- `styles.css`: visual styling.
- `app.js`: CSV loading, summary calculations, and SVG path rendering.
