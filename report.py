"""Unified report server for the PhD project.

Serves experiments dashboard, coverage reports, and project reports from one place.

Usage:
    uv run report.py              # Start server (default)
    uv run report.py --generate   # Generate reports without serving
"""

import argparse
import csv
import http.server
import json
import socketserver
import webbrowser
from pathlib import Path

# =============================================================================
# Data Loading
# =============================================================================


def load_registry(registry_path: Path) -> dict:
    """Load the experiments registry."""
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_epochs_csv(experiment_dir: Path) -> list[dict]:
    """Load epoch data from an experiment's CSV file."""
    csv_path = experiment_dir / "epochs.csv"
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_best_epoch_metrics(metrics: list[dict]) -> dict:
    """Extract metrics from the best epoch (lowest validation loss).

    Returns metrics from the epoch marked with is_best=1, or the epoch
    with the lowest valid_loss if no is_best marker exists.
    """
    if not metrics:
        return {}

    # Find the best epoch (marked with is_best=1)
    best_row = None
    for row in metrics:
        if row.get("is_best") == "1":
            best_row = row

    # Fallback: find epoch with lowest valid_loss
    if best_row is None:
        min_loss = float("inf")
        for row in metrics:
            try:
                loss = float(row.get("valid_loss", "inf"))
                if loss < min_loss:
                    min_loss = loss
                    best_row = row
            except ValueError:
                pass

    if best_row is None:
        return {}

    # Extract metrics from best epoch
    result = {}
    for col in ["epoch", "valid_loss", "ssim", "ms_ssim", "mae", "gradient_mae", "psnr"]:
        val = best_row.get(col, "")
        if val != "":
            try:
                result[col] = float(val)
            except ValueError:
                pass

    return result


def format_value(val: str | float | None, precision: int = 4) -> str:
    """Format a numeric value for display."""
    if val == "" or val is None:
        return "-"
    try:
        num = float(val)
        if abs(num) < 0.0001 and num != 0:
            return f"{num:.2e}"
        return f"{num:.{precision}f}"
    except (ValueError, TypeError):
        return str(val)


# =============================================================================
# Index Page Generation
# =============================================================================


def generate_index_page(output_path: Path) -> None:
    """Generate the main index page with links to all reports."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhD Project Reports</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0; padding: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: white; margin-bottom: 10px; font-size: 2.5em; }
        .subtitle { color: rgba(255,255,255,0.8); margin-bottom: 40px; font-size: 1.1em; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            text-decoration: none;
            color: inherit;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 20px rgba(0,0,0,0.2);
        }
        .card h2 { margin: 0 0 10px 0; color: #333; }
        .card p { margin: 0; color: #666; font-size: 14px; line-height: 1.5; }
        .card .icon { font-size: 2em; margin-bottom: 15px; }
        .status {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            margin-top: 15px;
        }
        .status-ok { background: #d4edda; color: #155724; }
        .status-missing { background: #f8d7da; color: #721c24; }
        footer {
            text-align: center;
            margin-top: 40px;
            color: rgba(255,255,255,0.7);
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PhD Project Reports</h1>
        <p class="subtitle">CT Slice Interpolation - Training Dashboard</p>

        <div class="cards">
            <a href="experiments/train_nn1_cropped/experiments_report.html" class="card">
                <div class="icon">📊</div>
                <h2>Experiments</h2>
                <p>Training experiments dashboard with metrics, status, and configuration details.</p>
                <span class="status status-ok">Refresh to update</span>
            </a>

            <a href="htmlcov/index.html" class="card">
                <div class="icon">📈</div>
                <h2>Test Coverage</h2>
                <p>Code coverage report from pytest. See which modules need more tests.</p>
                <span class="status status-ok">Run pytest to update</span>
            </a>

            <a href="REPORT.md" class="card" target="_blank">
                <div class="icon">📋</div>
                <h2>Analysis Report</h2>
                <p>Project analysis, test gaps, and improvement recommendations.</p>
                <span class="status status-ok">Markdown</span>
            </a>

            <a href="README.md" class="card" target="_blank">
                <div class="icon">📖</div>
                <h2>Documentation</h2>
                <p>Project README with setup instructions and usage guide.</p>
                <span class="status status-ok">Markdown</span>
            </a>
        </div>

        <footer>
            Refresh browser to update reports
        </footer>
    </div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html)


# =============================================================================
# Experiments Report Generation
# =============================================================================


def generate_experiments_report(registry_path: Path, experiments_dir: Path, output_path: Path) -> None:
    """Generate the experiments HTML report."""
    registry = load_registry(registry_path)

    experiments_data = []
    for exp_name, exp_info in registry.items():
        exp_dir = experiments_dir / exp_name
        metrics = load_epochs_csv(exp_dir)
        best_metrics = get_best_epoch_metrics(metrics)
        experiments_data.append(
            {
                "name": exp_name,
                "info": exp_info,
                "metrics": metrics,
                "best": best_metrics,
            }
        )

    experiments_data.sort(key=lambda x: x["best"].get("valid_loss", float("inf")))
    html = _generate_experiments_html(experiments_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html)

    print(f"Experiments report: {output_path} ({len(experiments_data)} experiments)")


def _generate_experiments_html(experiments_data: list[dict]) -> str:
    """Generate the experiments HTML content."""
    summary_rows = []
    if not experiments_data:
        summary_rows.append("""
        <tr>
            <td colspan="9" style="text-align: center; padding: 40px; color: #666;">
                <div style="font-size: 18px; margin-bottom: 10px;">No experiments yet</div>
                <div style="font-size: 14px;">Run <code>uv run train.py</code> to start training</div>
            </td>
        </tr>
        """)

    for i, exp in enumerate(experiments_data):
        info = exp["info"]
        best = exp["best"]
        config = info.get("config", {})

        # Extract nested config values
        loss_name = config.get("loss", {}).get("name", "N/A") if isinstance(config.get("loss"), dict) else "N/A"

        status_class = {
            "FINISHED_EPOCHS": "status-finished",
            "EARLY_STOPPING": "status-early",
            "RUNNING": "status-running",
            "ERROR": "status-error",
            "NOT_STARTED": "status-pending",
        }.get(info.get("status", ""), "")

        # Best epoch number (from metrics, not final_epoch)
        best_epoch = int(best.get("epoch", 0)) if best.get("epoch") else "-"

        row = f"""
        <tr class="experiment-row" onclick="toggleDetails({i})">
            <td><span class="expand-icon" id="icon-{i}">▶</span> {exp["name"]}</td>
            <td><span class="status {status_class}">{info.get("status", "N/A")}</span></td>
            <td>{loss_name}</td>
            <td>{format_value(best.get("valid_loss"))}</td>
            <td>{format_value(best.get("ssim"))}</td>
            <td>{format_value(best.get("ms_ssim"))}</td>
            <td>{format_value(best.get("mae"))}</td>
            <td>{format_value(best.get("psnr"))}</td>
            <td>{best_epoch}</td>
        </tr>
        """
        summary_rows.append(row)
        summary_rows.append(_generate_details_row(i, exp))

    summary_table = "\n".join(summary_rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiments Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5; color: #333;
        }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
        h1 {{ margin: 0; }}
        .subtitle {{ color: #666; margin: 5px 0 0 0; }}
        .nav-btn {{
            padding: 10px 20px; background: #667eea; color: white;
            border: none; border-radius: 6px; text-decoration: none; font-size: 14px;
        }}
        .nav-btn:hover {{ background: #5a6fd6; }}
        .summary-stats {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .stat-card {{
            background: white; padding: 15px 25px; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-card .number {{ font-size: 24px; font-weight: bold; color: #667eea; }}
        .stat-card .label {{ color: #666; font-size: 14px; }}
        table {{
            width: 100%; border-collapse: collapse; background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden;
        }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #555; position: sticky; top: 0; cursor: pointer; user-select: none; }}
        th:hover {{ background: #e8e8e8; }}
        th.sort-asc::after {{ content: " ▲"; font-size: 10px; }}
        th.sort-desc::after {{ content: " ▼"; font-size: 10px; }}
        .experiment-row {{ cursor: pointer; transition: background 0.2s; }}
        .experiment-row:hover {{ background: #f0f7ff; }}
        .expand-icon {{ display: inline-block; width: 20px; transition: transform 0.2s; }}
        .expand-icon.expanded {{ transform: rotate(90deg); }}
        .details-row {{ display: none; }}
        .details-row.visible {{ display: table-row; }}
        .details-content {{ background: #fafafa; padding: 20px; }}
        .metrics-table {{ font-size: 13px; margin-top: 10px; }}
        .metrics-table th {{ background: #e8e8e8; cursor: pointer; user-select: none; }}
        .metrics-table th:hover {{ background: #d0d0d0; }}
        .metrics-table th.sort-asc::after {{ content: " ▲"; font-size: 10px; }}
        .metrics-table th.sort-desc::after {{ content: " ▼"; font-size: 10px; }}
        .metrics-table td {{ padding: 8px 12px; }}
        .metrics-table .best-row {{ background: #e8f5e9 !important; font-weight: 600; }}
        .status {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; }}
        .status-finished {{ background: #e8f5e9; color: #2e7d32; }}
        .status-early {{ background: #fff3e0; color: #ef6c00; }}
        .status-running {{ background: #e3f2fd; color: #1565c0; }}
        .status-error {{ background: #ffebee; color: #c62828; }}
        .status-pending {{ background: #f5f5f5; color: #757575; }}
        .config-grid {{
            display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px; margin-bottom: 15px;
        }}
        .config-item {{ background: #f0f0f0; padding: 8px 12px; border-radius: 4px; font-size: 13px; }}
        .config-item .key {{ color: #666; }}
        .config-item .value {{ font-weight: 500; }}
        .tabs {{ display: flex; gap: 5px; margin-bottom: 10px; }}
        .tab {{
            padding: 8px 16px; background: #e0e0e0; border: none;
            border-radius: 4px 4px 0 0; cursor: pointer; font-size: 13px;
        }}
        .tab.active {{ background: white; font-weight: 500; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Experiments Report</h1>
            <p class="subtitle">CT Slice Interpolation Training</p>
        </div>
        <a href="/index.html" class="nav-btn">← Back to Dashboard</a>
    </div>

    <div class="summary-stats">
        <div class="stat-card">
            <div class="number">{len(experiments_data)}</div>
            <div class="label">Total</div>
        </div>
        <div class="stat-card">
            <div class="number">{sum(1 for e in experiments_data if e["info"].get("status") == "RUNNING")}</div>
            <div class="label">Running</div>
        </div>
        <div class="stat-card">
            <div class="number">{sum(1 for e in experiments_data if e["info"].get("status") == "NOT_STARTED")}</div>
            <div class="label">Queued</div>
        </div>
        <div class="stat-card">
            <div class="number">{sum(1 for e in experiments_data if e["info"].get("status") in ("FINISHED_EPOCHS", "EARLY_STOPPING"))}</div>
            <div class="label">Completed</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th onclick="sortMainTable(0)">Experiment</th>
                <th onclick="sortMainTable(1)">Status</th>
                <th onclick="sortMainTable(2)">Loss Fn</th>
                <th onclick="sortMainTable(3)">Val Loss</th>
                <th onclick="sortMainTable(4)">SSIM</th>
                <th onclick="sortMainTable(5)">MS-SSIM</th>
                <th onclick="sortMainTable(6)">MAE</th>
                <th onclick="sortMainTable(7)">PSNR</th>
                <th onclick="sortMainTable(8)">Best Ep</th>
            </tr>
        </thead>
        <tbody>
            {summary_table}
        </tbody>
    </table>

    <script>
        function toggleDetails(index) {{
            const row = document.getElementById('details-' + index);
            const icon = document.getElementById('icon-' + index);
            row.classList.toggle('visible');
            icon.classList.toggle('expanded');
        }}
        function switchTab(index, tabName) {{
            document.querySelectorAll('#details-' + index + ' .tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('#details-' + index + ' .tab').forEach(t => t.classList.remove('active'));
            document.getElementById('tab-' + tabName + '-' + index).classList.add('active');
            event.target.classList.add('active');
        }}
        function sortMainTable(colIndex) {{
            const table = document.querySelector('body > table');
            const tbody = table.querySelector('tbody');
            const th = table.querySelectorAll('thead th')[colIndex];
            const isAsc = th.classList.contains('sort-asc');

            // Clear sort indicators
            table.querySelectorAll('thead th').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));

            // Collect row pairs (experiment row + details row) - direct children only
            const allRows = Array.from(tbody.querySelectorAll(':scope > tr'));
            const pairs = [];
            for (let i = 0; i < allRows.length; i += 2) {{
                pairs.push([allRows[i], allRows[i + 1]]);
            }}

            pairs.sort((a, b) => {{
                const aText = a[0].cells[colIndex].textContent.trim();
                const bText = b[0].cells[colIndex].textContent.trim();
                const aNum = parseFloat(aText);
                const bNum = parseFloat(bText);
                const aIsNum = !isNaN(aNum) && aText !== '-';
                const bIsNum = !isNaN(bNum) && bText !== '-';

                // Push dashes to the end
                if (!aIsNum && !bIsNum) return aText.localeCompare(bText) * (isAsc ? -1 : 1);
                if (!aIsNum) return 1;
                if (!bIsNum) return -1;

                return isAsc ? bNum - aNum : aNum - bNum;
            }});

            th.classList.add(isAsc ? 'sort-desc' : 'sort-asc');
            pairs.forEach(([expRow, detailRow]) => {{
                tbody.appendChild(expRow);
                tbody.appendChild(detailRow);
            }});
        }}
        function sortTable(table, colIndex) {{
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const th = table.querySelectorAll('th')[colIndex];
            const isAsc = th.classList.contains('sort-asc');

            // Clear sort indicators from all headers
            table.querySelectorAll('th').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));

            // Sort rows
            rows.sort((a, b) => {{
                const aVal = a.cells[colIndex].textContent.trim();
                const bVal = b.cells[colIndex].textContent.trim();
                const aNum = parseFloat(aVal);
                const bNum = parseFloat(bVal);

                // Handle numeric vs string comparison
                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return isAsc ? bNum - aNum : aNum - bNum;
                }}
                return isAsc ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
            }});

            // Update sort indicator
            th.classList.add(isAsc ? 'sort-desc' : 'sort-asc');

            // Reorder rows
            rows.forEach(row => tbody.appendChild(row));
        }}
    </script>
</body>
</html>
"""


def _generate_details_row(index: int, exp: dict) -> str:
    """Generate the expandable details row for an experiment."""
    info = exp["info"]
    config = info.get("config", {})
    metrics = exp["metrics"]

    def item(k: str, v: str | int) -> str:
        return f'<div class="config-item"><span class="key">{k}:</span> <span class="value">{v}</span></div>'

    config_items = [item(k, v) for k, v in config.items()]
    config_items.append(item("created", info.get("created", "N/A")))
    config_items.append(item("finished", info.get("finished", "N/A")))
    config_items.append(item("runs", info.get("runs", 1)))

    if metrics:
        columns = list(metrics[0].keys())
        header_cells = "".join(
            f'<th onclick="sortTable(this.closest(\'table\'), {i})">{col}</th>' for i, col in enumerate(columns)
        )
        best_epochs = {int(row["epoch"]) for row in metrics if row.get("is_best") == "1"}
        metric_rows = []
        for row in reversed(metrics):  # Latest epoch first
            epoch = int(row["epoch"])
            row_class = "best-row" if epoch in best_epochs else ""
            cells = "".join(f"<td>{format_value(row.get(col, ''))}</td>" for col in columns)
            metric_rows.append(f'<tr class="{row_class}">{cells}</tr>')
        tbody = "".join(metric_rows)
        metrics_table = (
            f'<table class="metrics-table"><thead><tr>{header_cells}</tr></thead><tbody>{tbody}</tbody></table>'
        )
    else:
        metrics_table = "<p>No metrics data available.</p>"

    return f"""
    <tr id="details-{index}" class="details-row">
        <td colspan="9">
            <div class="details-content">
                <div class="tabs">
                    <button class="tab active" onclick="switchTab({index}, 'config')">Configuration</button>
                    <button class="tab" onclick="switchTab({index}, 'metrics')">Full Metrics</button>
                </div>
                <div id="tab-config-{index}" class="tab-content active">
                    <div class="config-grid">{chr(10).join(config_items)}</div>
                </div>
                <div id="tab-metrics-{index}" class="tab-content">{metrics_table}</div>
            </div>
        </td>
    </tr>
    """


# =============================================================================
# Server
# =============================================================================


class ReusableTCPServer(socketserver.TCPServer):
    """TCPServer that allows port reuse."""

    allow_reuse_address = True


def create_handler(
    registry_path: Path,
    experiments_dir: Path,
    experiments_report: Path,
    index_page: Path,
) -> type:
    """Create a custom HTTP handler that regenerates reports on demand."""

    class OnDemandReportHandler(http.server.SimpleHTTPRequestHandler):
        """HTTP handler that regenerates reports when requested."""

        def do_GET(self) -> None:
            """Handle GET requests, regenerating reports on demand."""
            # Regenerate index.html on request
            if self.path in ("/", "/index.html"):
                try:
                    generate_index_page(index_page)
                except Exception as e:
                    print(f"Error regenerating index: {e}")

            # Regenerate experiments report on request
            elif self.path.endswith("experiments_report.html"):
                try:
                    generate_experiments_report(registry_path, experiments_dir, experiments_report)
                except Exception as e:
                    print(f"Error regenerating experiments report: {e}")

            # Serve the file
            super().do_GET()

        def log_message(self, format: str, *args: object) -> None:
            """Suppress default logging for cleaner output."""
            pass

    return OnDemandReportHandler


def serve(
    registry_path: Path,
    experiments_dir: Path,
    experiments_report: Path,
    index_page: Path,
    port: int = 0,
    open_browser: bool = True,
) -> None:
    """Start the report server."""
    # Generate initial reports
    generate_index_page(index_page)
    generate_experiments_report(registry_path, experiments_dir, experiments_report)

    # Create handler that regenerates on demand
    handler = create_handler(registry_path, experiments_dir, experiments_report, index_page)

    with ReusableTCPServer(("", port), handler) as httpd:
        actual_port = httpd.server_address[1]
        url = f"http://localhost:{actual_port}/index.html"

        print(f"\n{'=' * 50}")
        print("  Report Server Running")
        print(f"{'=' * 50}")
        print(f"\n  Dashboard:   {url}")
        print(f"  Experiments: http://localhost:{actual_port}/{experiments_report}")
        print(f"  Coverage:    http://localhost:{actual_port}/htmlcov/index.html")
        print("\n  Reports regenerate on browser refresh")
        print("  Press Ctrl+C to stop\n")

        if open_browser:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the report server."""
    parser = argparse.ArgumentParser(
        description="PhD Project Report Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run report.py              # Start server and open browser
  uv run report.py --no-browser # Start server without opening browser
  uv run report.py --generate   # Generate reports without serving
  uv run report.py --port 8000  # Use specific port
        """,
    )
    parser.add_argument("--generate", action="store_true", help="Generate reports without serving")
    parser.add_argument("--port", type=int, default=0, help="Port (default: random)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    registry_path = Path("./experiments/experiments_registry.json")
    experiments_dir = Path("./experiments/train_nn1_cropped")
    experiments_report = Path("./experiments/train_nn1_cropped/experiments_report.html")
    index_page = Path("./index.html")

    if args.generate:
        generate_index_page(index_page)
        generate_experiments_report(registry_path, experiments_dir, experiments_report)
        print(f"Index page: {index_page}")
    else:
        serve(registry_path, experiments_dir, experiments_report, index_page, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
