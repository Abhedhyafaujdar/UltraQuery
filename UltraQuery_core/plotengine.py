import time

import matplotlib.pyplot as plt
import pandas as pd
import os
import textwrap

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333F4B'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#A0A0A0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.7
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlepad'] = 15
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.color'] = '#333F4B'
plt.rcParams['ytick.color'] = "#303C48"
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

x = time.perf_counter()


class UltraQuery_plot:
    def __init__(self, file, x, y):
        self.x = x.strip()
        self.y = y.strip()

        file_path = os.path.abspath(file)
        if not os.path.exists(file_path):
            print(f"[❌] File not found: {file_path}")
            exit(1)

        # use encoding that won't crash on Windows CSVs
        self.file = pd.read_csv(file_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
        self.file.columns = self.file.columns.str.strip()

        print(f"[✓] Columns: {self.file.columns.tolist()}")

        if self.x not in self.file.columns:
            print(f"[✗] Column '{self.x}' not found.")
            exit(1)
        if self.y not in self.file.columns:
            print(f"[✗] Column '{self.y}' not found.")
            exit(1)

        # Data size detection
        self.num_rows = len(self.file)
        self.unique_x = self.file[self.x].nunique()
        self.is_large_data = self.num_rows > 10000 or self.unique_x > 50
        self.sampled = False

        if self.is_large_data:
            print(f"[⚠] Large dataset detected ({self.num_rows} rows, {self.unique_x} unique {self.x}). Sampling for performance.")
            if self.num_rows > 10000:
                self.file = self.file.sample(n=10000, random_state=42)
                self.sampled = True
                print("[✓] Sampled 10,000 rows for plotting.")

        self.counts = self.file[self.x].value_counts()
        if pd.api.types.is_numeric_dtype(self.file[self.y]):
            self.values = self.file.groupby(self.x, sort=False)[self.y].sum()
        else:
            self.values = self.counts

    def _bar(self):
        values = self.values
        bars = plt.bar(
            values.index,
            values.values,
            color=plt.cm.viridis_r(values.values / max(values.values)),
            alpha=0.85,
            edgecolor='#222222',
            linewidth=0.8
        )
        plt.xlabel(self.x, fontsize=14)
        plt.ylabel(self.y, fontsize=14)
        plt.title(f'{self.x} vs {self.y}', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.6)
        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.show()

    def _pie(self):
        # Aggregate if too many categories
        if len(self.counts) > 20:
            top_counts = self.counts.nlargest(20)
            others_sum = self.counts.iloc[20:].sum()
            if others_sum > 0:
                top_counts['Others'] = others_sum
            self.counts = top_counts
            print("[⚠] Aggregated categories: showing top 20 and 'Others' for readability.")

        plt.pie(
            self.counts.values,
            labels=self.counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=plt.cm.plasma(self.counts.values / max(self.counts.values)),
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.7},
            textprops={'fontsize': 12, 'color': 'black'}
        )
        plt.title(f'Market Share by {self.x}', fontsize=16)
        plt.tight_layout()
        plt.show()

    def _line(self):
        x_vals = range(len(self.values.index))
        y_vals = self.values.values
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', color="#1575ba", alpha=0.85, linewidth=2)
        plt.xticks(x_vals, self.values.index, rotation=45, ha='right', fontsize=12)
        plt.xlabel(self.x, fontsize=14)
        plt.ylabel(self.y, fontsize=14)
        plt.title(f'{self.x} vs {self.y}', fontsize=16)
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.show()

    def _scatter(self):
        values = self.values
        plt.scatter(
            range(len(values.index)),
            values.values,
            s=100,
            c=plt.cm.cividis(values.values / max(values.values)),
            alpha=0.85,
            edgecolors='black',
            linewidth=0.7
        )
        plt.xlabel(self.x, fontsize=14)
        plt.ylabel(self.y, fontsize=14)
        plt.title(f'{self.x} vs {self.y}', fontsize=16)
        plt.xticks(range(len(values.index)), values.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.show()

    def _histogram(self):
        # Dynamic bins based on data size
        bins = min(50, len(self.counts) // 100 + 10)
        plt.hist(
            self.counts.values,
            bins=bins,
            edgecolor='black',
            color=plt.cm.magma(0.7),
            alpha=0.85
        )
        plt.xlabel(self.x, fontsize=14)
        plt.ylabel(self.y, fontsize=14)
        plt.title(f'Distribution of {self.x}', fontsize=16)
        plt.grid(True, axis='y', alpha=0.6)
        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.show()

    def plot(self, plot_type):
        plot_methods = {
            'bar': self._bar,
            'pie': self._pie,
            'line': self._line,
            'scatter': self._scatter,
            'histogram': self._histogram
        }
        if plot_type not in plot_methods:
            raise ValueError(f"Unsupported plot type: {plot_type}. Available: {', '.join(plot_methods.keys())}")
        plot_methods[plot_type]()


def _load_plot_data(file):
    file_path = os.path.abspath(file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = pd.read_csv(file_path, encoding='ISO-8859-1', engine='python', on_bad_lines='skip')
    data.columns = data.columns.str.strip()
    if data.empty or len(data.columns) == 0:
        raise ValueError("CSV has no rows or columns")
    return data


def infer_schema(data):
    """Infer chart-friendly roles from column names and values."""
    names = {column: str(column).lower().replace('-', '_').replace(' ', '_') for column in data.columns}
    numeric = data.select_dtypes(include='number').columns.tolist()
    ignored_numeric = {'index', 'id', 'model', 'year', 'timestamp'}
    ignored_fragments = ('lat', 'lon', 'time', 'date', 'scan', 'track', 'version', 'index', 'id')
    metric_words = ('frp', 'sales', 'revenue', 'amount', 'price', 'brightness', 'value', 'score', 'count', 'total', 'quantity', 'duration')
    metrics = [column for column in numeric
               if names[column] not in ignored_numeric
               and not any(fragment in names[column] for fragment in ignored_fragments)]
    metrics.sort(key=lambda column: (
        next((index for index, word in enumerate(metric_words) if word in names[column]), len(metric_words)),
        names[column],
    ))

    date_columns = []
    for column in data.columns:
        name = names[column]
        if any(word in name for word in ('date', 'time', 'timestamp', 'year', 'month')):
            converted = pd.to_datetime(data[column], errors='coerce')
            if converted.notna().mean() >= 0.6:
                date_columns.append(column)

    label_words = ('event', 'type', 'category', 'product', 'name', 'user', 'customer', 'region', 'country', 'title', 'satellite', 'daynight')
    label_columns = [column for column in data.columns
                     if not pd.api.types.is_numeric_dtype(data[column])
                     and column not in date_columns
                     and not names[column] in {'id', 'user_id', 'customer_id', 'url'}
                     and not names[column] in {'url', 'description', 'raw_description'}
                     and data[column].nunique(dropna=True) <= max(50, len(data) * 0.8)]
    label_columns.sort(key=lambda column: (
        next((index for index, word in enumerate(label_words) if word in names[column]), len(label_words)),
        names[column],
    ))

    return {
        'date': date_columns[0] if date_columns else None,
        'label': label_columns[0] if label_columns else None,
        'metric': metrics[0] if metrics else None,
        'metrics': metrics,
        'recommended_chart': 'line' if date_columns and metrics else 'bar',
    }


def auto_plot(file, plot_type='bar'):
    """Choose a useful chart, category/date column, and numeric metric automatically."""
    data = _load_plot_data(file)
    schema = infer_schema(data)
    if not schema['metric']:
        raise ValueError("Automatic plotting needs at least one meaningful numeric column")
    x_column = schema['date'] or schema['label'] or data.columns[0]
    chart_type = schema['recommended_chart'] if plot_type == 'auto' else plot_type
    UltraQuery_plot(file, x_column, schema['metric']).plot(chart_type)


def comparison_plot(file, x_column=None, y_columns=None):
    """Create a grouped comparison chart for multiple numeric columns."""
    data = _load_plot_data(file)
    numeric_columns = data.select_dtypes(include='number').columns.tolist()
    if y_columns:
        requested = [column.strip() for column in y_columns.split(',') if column.strip()]
        missing = [column for column in requested if column not in numeric_columns]
        if missing:
            raise ValueError(f"Comparison columns must be numeric: {', '.join(missing)}")
        numeric_columns = requested
    if len(numeric_columns) < 2:
        raise ValueError("Comparison plotting needs at least two numeric columns")

    if x_column is None:
        categorical = [column for column in data.columns if column not in numeric_columns]
        x_column = categorical[0] if categorical else None
    if x_column and x_column in data.columns:
        chart_data = data.groupby(x_column, sort=False)[numeric_columns].sum()
        chart_data.plot(kind='bar', figsize=(10, 6), colormap='viridis', edgecolor='#222222')
        plt.xlabel(x_column)
    else:
        chart_data = data[numeric_columns]
        chart_data.plot(kind='bar', figsize=(10, 6), colormap='viridis', edgecolor='#222222')
        plt.xlabel('Rows')
    plt.ylabel('Value')
    plt.title('Column comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.6)
    plt.tight_layout()
    plt.show()


def multi_dataset_plot(files, metric=None, output=None):
    """Compare the mean of a common numeric metric across CSV datasets."""
    summaries = []
    labels = []
    for file in files:
        data = _load_plot_data(file)
        numeric = data.select_dtypes(include='number')
        if numeric.empty:
            raise ValueError(f"Dataset has no numeric columns: {file}")
        summaries.append(numeric.mean())
        labels.append(os.path.basename(file))

    combined = pd.DataFrame(summaries, index=labels)
    if metric:
        if metric not in combined.columns:
            raise ValueError(f"Numeric metric not found in all datasets: {metric}")
        combined = combined[[metric]]
    else:
        common = combined.columns[combined.notna().all()].tolist()
        if not common:
            raise ValueError("Datasets do not share a numeric metric")
        combined = combined[common]

    axis = combined.plot(kind='bar', figsize=(10, 6), colormap='plasma', edgecolor='#222222')
    plt.xlabel('Dataset')
    plt.ylabel('Mean value')
    plt.title('Dataset comparison')
    plt.xticks(rotation=35, ha='right')
    plt.grid(True, axis='y', alpha=0.6)
    plt.tight_layout()
    figure = axis.get_figure()
    if output:
        figure.savefig(output, dpi=160, bbox_inches='tight')
        print(f'[+] Comparison chart saved to {os.path.abspath(output)}')
    else:
        plt.show()
    plt.close(figure)


def smart_dashboard(file, scraped_pages=None, ai_summary=None, output=None):
    """Render a polished, data-driven dashboard for a CSV dataset."""
    data = _load_plot_data(file)
    numeric = data.select_dtypes(include='number').copy()
    numeric = numeric.drop(columns=[column for column in numeric.columns
                                    if column.lower() in {'index', 'id', 'model'}], errors='ignore')
    if numeric.empty:
        raise ValueError("Smart dashboard needs at least one meaningful numeric column")

    plt.style.use('dark_background')
    background = '#0b1020'
    panel = '#121a2d'
    accent = '#45e0c2'
    orange = '#ffb86b'
    muted = '#9aa8c2'
    figure = plt.figure(figsize=(16, 10), facecolor=background)
    grid = figure.add_gridspec(3, 2, height_ratios=[0.7, 1.6, 1.6], hspace=0.38, wspace=0.22)

    title_axis = figure.add_subplot(grid[0, :])
    title_axis.set_facecolor(background)
    title_axis.axis('off')
    title_axis.text(0.01, 0.95, 'ULTRAQUERY // SMART DATA INTELLIGENCE', color=accent,
                    fontsize=11, fontweight='bold', family='DejaVu Sans Mono')
    title_axis.text(0.01, 0.12, os.path.basename(file), color='white', fontsize=19, fontweight='bold')
    title_axis.text(0.99, 0.12, f'{len(data):,} records  |  {len(data.columns)} fields',
                    color=muted, fontsize=11, ha='right')

    schema = infer_schema(data)
    numeric_names = numeric.columns.tolist()
    primary = schema['metric'] if schema['metric'] in numeric.columns else numeric_names[0]
    cards = [
        ('ROWS', f'{len(data):,}', accent),
        ('PRIMARY MEAN', f'{numeric[primary].mean():,.2f}', orange),
        ('PRIMARY MIN', f'{numeric[primary].min():,.2f}', '#8be9fd'),
        ('PRIMARY MAX', f'{numeric[primary].max():,.2f}', '#ff79c6'),
    ]
    for index, (label, value, color) in enumerate(cards):
        axis = figure.add_axes((0.03 + index * 0.235, 0.805, 0.205, 0.075))
        axis.set_facecolor(panel)
        axis.axis('off')
        axis.text(0.08, 0.68, label, color=muted, fontsize=8, fontweight='bold')
        axis.text(0.08, 0.18, value, color=color, fontsize=18, fontweight='bold')

    first = figure.add_subplot(grid[1, 0], facecolor=panel)
    top = data.nlargest(min(12, len(data)), primary)
    label_column = schema['label']
    labels = top[label_column].astype(str) if label_column else top.index.astype(str)
    labels = labels.map(lambda value: value.encode('ascii', 'replace').decode('ascii'))
    first.barh(labels[::-1], top[primary].iloc[::-1], color=accent, alpha=0.9)
    first.set_title(f'TOP {len(top)} BY {primary.upper()}', loc='left', color='white', fontweight='bold')
    first.tick_params(axis='y', labelsize=8, colors=muted)
    first.tick_params(axis='x', colors=muted)
    first.grid(axis='x', alpha=0.15)

    second = figure.add_subplot(grid[1, 1], facecolor=panel)
    if len(numeric_names) >= 2:
        x_name, y_name = numeric_names[:2]
        second.scatter(numeric[x_name], numeric[y_name], c=numeric[primary], cmap='magma',
                       s=45, alpha=0.8, edgecolors='none')
        second.set_xlabel(x_name, color=muted)
        second.set_ylabel(y_name, color=muted)
        second.set_title('METRIC RELATIONSHIP', loc='left', color='white', fontweight='bold')
    else:
        second.hist(numeric[primary].dropna(), bins=min(20, max(5, len(data) // 5)),
                    color=orange, alpha=0.9)
        second.set_title(f'{primary.upper()} DISTRIBUTION', loc='left', color='white', fontweight='bold')
    second.tick_params(colors=muted)
    second.grid(alpha=0.15)

    third = figure.add_subplot(grid[2, 0], facecolor=panel)
    availability = next((column for column in data.columns if column.lower() in {'availability', 'status', 'state'}), None)
    if availability:
        counts = data[availability].fillna('Unknown').astype(str).value_counts()
        third.pie(counts.values, labels=counts.index, autopct='%1.0f%%', startangle=90,
                  colors=['#45e0c2', '#ff6b8a', '#ffb86b', '#8be9fd'],
                  textprops={'color': 'white', 'fontsize': 9})
        third.set_title(f'{availability.upper()} MIX', loc='left', color='white', fontweight='bold')
    else:
        correlation = numeric.corr()[primary].drop(primary).sort_values() if len(numeric_names) > 1 else pd.Series(dtype=float)
        if correlation.empty:
            third.text(0.08, 0.5, 'Add a status column\nfor availability insights.', color=muted, fontsize=12)
        else:
            correlation.plot(kind='barh', ax=third, color=accent)
            third.set_title(f'CORRELATION WITH {primary.upper()}', loc='left', color='white', fontweight='bold')
            third.tick_params(colors=muted)
            third.grid(axis='x', alpha=0.15)
    third.set_facecolor(panel)

    fourth = figure.add_subplot(grid[2, 1], facecolor=panel)
    fourth.axis('off')
    fourth.text(0.02, 0.92, 'EVIDENCE + AI SIGNALS', color='white', fontsize=11, fontweight='bold')
    evidence = []
    if scraped_pages:
        successful = sum(1 for page in scraped_pages if page.get('status') == 200)
        evidence.append(f'WebCmd pages fetched: {len(scraped_pages)} ({successful} successful)')
        for page in scraped_pages[:3]:
            evidence.append(f"- {page.get('title') or page.get('url', 'page')}")
    else:
        evidence.append('No URL evidence attached to this run.')
    if ai_summary:
        evidence.append('')
        evidence.append('AI summary:')
        evidence.extend(textwrap.wrap(str(ai_summary), width=62)[:7])
    for line_index, line in enumerate(evidence):
        fourth.text(0.02, 0.78 - line_index * 0.095, line, color=accent if line.startswith('-') else muted,
                    fontsize=9, va='top')

    if output:
        figure.savefig(output, dpi=160, facecolor=background, bbox_inches='tight')
        print(f'[+] Dashboard saved to {os.path.abspath(output)}')
    else:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    y = time.perf_counter()
    available_types = ['bar', 'pie', 'line', 'scatter', 'histogram']
    print("Available plot types:", ", ".join(available_types))
    user_choice = input("Choose plot type: ").strip().lower()
    uq = UltraQuery_plot("cars.csv", "Engines", "CC/Battery Capacity")
    try:
        uq.plot(user_choice)
    except ValueError as e:
        print(e)

    print(f"Execution time: {y - x:.4f} seconds")
