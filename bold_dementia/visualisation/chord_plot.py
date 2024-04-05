from pycirclize import Circos
import math

def make_links(matrix_df):
    links = []

    for source, row in matrix_df.iterrows():
        for target_rsn, val in row.items():
            if math.isclose(val, 0):
                continue
            elif val > 0:
                colour = "red"
            elif val < 0:
                colour = "blue"

            offset = abs((10 * val))
            assert offset < 10
            print(source, target_rsn, val, offset)
            if source != target_rsn:
                links.append(
                    (
                        (source, 5 - offset / 2, 5 + offset / 2),
                        (target_rsn, 5 - offset / 2, 5 + offset / 2),
                        colour
                    )
                )
            else:
                links.append(
                    (
                        (source, 0, offset),
                        (target_rsn, 10 - offset, 10),
                        colour
                    )
                )
    return links
    

def create_chord(networks, links, **sector_kws):
    sectors = {n: 10 for n in networks}
    circos = Circos(sectors, space=1)
    
    r_lim=(90, 100)
    for sector in circos.sectors:
        # Plot label, outer track axis & xticks
        sector.text(sector.name, **sector_kws)
        outer_track = sector.add_track(r_lim)
        color = "gray"
        outer_track.axis(fc=color, alpha=0.5)
    
    
    # Plot links
    for link in links:
        circos.link(*link[:2], color=link[2])
    #circos.link(('RSN01', 1, 3), ('RSN01', 6, 9), color='blue')
    
    
    
    fig = circos.plotfig()
    return fig
    