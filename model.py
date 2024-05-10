from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)

# sample data in lat and long of famous places
places = np.array([
    [7.293757981, 80.64133572],  # Sri Dalada Maligawa, Kandy
    [7.268476644, 80.59655809],  # Royal Botanical Garden, Kandy
    [7.293194044, 80.64291027],  # British Garrison Cemetery, Kandy
    [7.291674725, 80.64040021],  # Kandy Lake, Kandy
    [7.054998318, 80.70525697],  # Ramboda Falls, Kandy
    [7.299750828, 80.64085834],  # Udawattakele Forest Reserve, Kandy
    [7.218125338, 80.5677111],  # Embekke Dewalaya, Kandy
    [7.161157314, 80.54727013],  # Ambuluwawa Tower, Kandy
    [7.234243761, 80.56492667],  # Sri Lankathilaka Rajamaha Viharaya, Kandy
    [7.268843275, 80.63515301],  # Hanthana International Bird Park & Recreation Center, Kandy
    [7.077731733, 80.50345734],  # Kabaragala, Kandy
    [7.252961239, 80.50423584],  # Kadugannawa Rock View and Rest Area, Kandy
    [7.075654427, 80.49089783],  # Raksagala - Kinihira Kanda, Kandy
    [7.102692608, 80.54013779],  # Cow Down Waterfall, Kandy
    [7.304778698, 80.55995487],  # Nelligala International Buddhist Centre, Kandy
    [7.232466471, 80.63825277],  # Katusu Konda කටුසු කොන්ද, Kandy
    [7.239861814, 80.78761103],  # Adikaramgama Victoria Dam View Point, Kandy
    [7.374990495, 80.78360411],  # Rangala Natural Pool, Kandy
    [7.38435714, 80.87089768],  # Kalugala Gerandi Ella Waterfall, Kandy
    [7.480518056, 80.6278836],  # Sindakatti Sri Kumaran Kovil, Matale
    [7.493872602, 80.6092815],  # Matale Town - View Point, Matale
    [7.682156212, 80.61398475],  # Nalanda Dam & Reservoir, Matale
    [7.668583763, 80.65283922],  # Bowetenna Reservoir, Matale
    [7.580741177, 80.78430281],  # Thelgamu Oya, Matale
    [7.407353714, 80.70556208],  # Hunnas Paradise, Matale
    [7.495234682, 80.60547002],  # Wiltshire Mountain, Matale
    [7.695483869, 80.63911412],  # Arangala Mountain Peak, Matale
    [7.420176522, 80.68568476],  # Themali Khan Waterfall, Matale
    [7.491205062, 80.63652568],  # German Estate, Matale
    [7.463730362, 80.70300655],  # Pitakanda Peak, Matale
    [7.524375629, 80.74894634],  # Rathhinda Ella Waterfall, Matale
    [7.593181418, 80.56957036],  # Hulangala Mini Worlds, Matale
    [7.856864975, 80.64838481],  # Dambulla Royal Cave Temple, Matale
    [7.554488422, 80.74090588],  # Pitawala Pathana Ella Fall, Matale
    [7.405175291, 80.6926918],  # Upper Hunnasfall Waterfall, Matale
    [7.953990904, 80.75229533],  # Sigiriya, Matale
    [7.967710474, 80.76216708],  # Pindurangala, Matale
    [7.915668037, 80.67964552],  # Enderagala Wana Senasuna Temple, Matale
    [7.879672114, 80.73525187],  # Kaludiya Pokuna, Matale
    [7.591676221, 80.75614093],  # Sera Ella Waterfall, Matale
    [7.564482591, 80.76735277],  # Wambatuthenna Waterfall, Matale
    [7.531138497, 80.7368184],  # Nalanda Gedige, Matale
    [7.560329265, 80.83222257],  # Kalu Ganga View Point
    [7.531190944, 80.73720949],  # Riverston
    [7.495442911, 80.69903322],  # Bambarakiri Ella
    [7.474382966, 80.7902062],  # Dumbara Ella Waterfall
    [7.497567143, 80.76767613],  # Walpolamulla (Smallest village in Sri Lanka)
    [7.435722442, 80.74923157],  # Gombaniya Hiking Trail (starting point)
    [7.437304963, 80.71149439],  # Kalabokka 360 Upper Division View Point, Matale
    [7.435964584, 80.69983457],  # Sembuwaththa, Matale
    [7.441579444, 80.68015581],  # One Tree Hill, Matale
    [7.452169319, 80.60950625],  # Padiwita Ambalama, Matale
    [7.479604433, 80.62409446],  # Sri Muththumari Amman Kovil, Matale
    [7.498122673, 80.6219905]  # Aluviharaya Rock Cave Temple, Matale
])

user_location = None
nearby_places = None


# KMeans Clustering model to find the nearest location from the reference point
def find_nearby_places(user_lat, user_lon):
    global nearby_places, user_location

    user_location = np.array([user_lat, user_lon])

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(places)

    cluster_labels = kmeans.predict(places)

    user_clusters = kmeans.predict(user_location.reshape(1, -1))[0]
    nearby_places_indices = np.where(cluster_labels == user_clusters)[0]

    nearby_places = places[nearby_places_indices]


# function to predict nearby places
@app.route('/predict', methods=['GET'])
def predict():
    if nearby_places is None:
        return jsonify({'message': 'No nearby places!'}), 400
    else:
        return jsonify(nearby_places.tolist())


# function to update the user's location
@app.route('/update_location', methods=['POST'])
def update_location():
    global user_location

    data = request.get_json()
    user_lat = float(data['lat'])  # Parse latitude from request
    user_lon = float(data['lon'])  # Parse longitude from request

    find_nearby_places(user_lat, user_lon)

    return jsonify({'message': 'Location updated successfully'})


# run the flask server.
if __name__ == '__main__':
    app.run(debug=True)
