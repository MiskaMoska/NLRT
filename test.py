from geographiclib.geodesic import Geodesic

geo = Geodesic.WGS84.Inverse(39.9042,116.4074,31.2304,121.4737)

distance = geo['s12']
print('distance: ', distance)
bearing = geo['azi1']
print('bearing at 1:', geo['azi1'])
print('bearing at 2:', geo['azi2'])