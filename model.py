from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)

# sample data in lat and long of famous places
places = np.array([
    [40.719074,  -74.050552], # Jersey City
    [40.7128, -74.0060],  # New York City
    [34.0522, -118.2437],  # Los Angeles
    [41.8781, -87.6298],  # Chicago
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
    user_lon= float(data['lon'])  # Parse longitude from request

    find_nearby_places(user_lat, user_lon)

    return jsonify({'message': 'Location updated successfully'})


# run the flask server.
if __name__ == '__main__':
    app.run(debug=True)
