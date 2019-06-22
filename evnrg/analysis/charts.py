import numpy as np
import pandas as pd
import seaborn as sns
import tempfile
import uuid


def catplot_ext(df: pd.DataFrame, **kwargs):
    sns.set(style="ticks")
    g = sns.catplot(
        **{'data': df, **kwargs}
    )

    g.set_xticklabels(df[x], rotation=30, ha="center")
    return g

def relplot_ext(df: pd.DataFrame, **kwargs):

    sns.set(style='ticks')
    sns.plt.xlim(df[x].min(), df[x].max())

    g = sns.relplot(
        **{'data': df, **kwargs}
    )

    
    g.set_xticklabels(df[x], rotation=30, ha="center")

    return g


def plot_line(df: pd.DataFrame,
    y: str, x: str, col: str, name: str, wrap: int=4, row=None, hue=None):

    sns.set(style='ticks')
    sns.plt.xlim(df[x].min(), df[x].max())

    g = sns.relplot(
        x=x,
        y=y,
        hue=hue,
        col=col,
        col_wrap=wrap,
        row=row,
        kind='line',
        data=df,
        facet_kws=dict(
            legend_out=True,
            margin_titles=True
        )
    )

    
    g.set_xticklabels(df[x], rotation=30, ha="center")

    return g



def plot_bar(df: pd.DataFrame, si: StorageInfo, 
    basepath: str, y: str, x: str, col: str, name: str, wrap: int=4, row=None):
    sns.set(style="ticks")
    g = sns.catplot(
        col=col,
        row=row,
        x=x,
        y=y,
        kind='bar',
        data=df,
        col_wrap=wrap,
        margin_titles=True,
        legend_out=True
    )

    g.set_xticklabels(df[x], rotation=30, ha="center")

    return upload_chart(
        fig=g,
        si=si,
        basepath=basepath,
        name=name
    )



def plot_facets(
    df: pd.DataFrame,
    si: StorageInfo,
    facet_opts: dict,
    map_func: callable,
    map_opts: dict,
    basepath: str,
    name: str,
    use_legend: bool = True,
    meta: dict = {}):

    current_palette = sns.color_palette()
    sns.palplot(current_palette)

    g = sns.FacetGrid(
        df,
        margin_titles=True,
        **facet_opts
    )

    g.map_dataframe(map_func, **map_opts)
    g.add_legend()

    dh = Storage(si)

    with tempfile.TemporaryDirectory() as tdir:
        basefile = tdir + '/' + uuid.uuid4().hex
        svgfile = basefile + '.svg'
        g.savefig(svgfile, dpi=300)

        out = dh.upload_file(
            svgfile,
            basepath + '/' + name + '.svg',
            'svg',
            meta=meta
        )
        return out      

    return {}





