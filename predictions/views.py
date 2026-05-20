import pickle
import os
from django.http import JsonResponse
from predictions.decision_tree import predict, bin_time, bin_speed, bin_weather, bin_light, bin_road

# Load tree once at module level
TREE_PATH = os.path.join(os.path.dirname(__file__), "trained_tree.pkl")
with open(TREE_PATH, "rb") as f:
    TRAINED_TREE = pickle.load(f)


def predict_risk(request):
    """
    API endpoint for risk prediction.
    GET /api/predict/?lat=53.80&lon=-1.55&time=17&day=Friday&weather=rain&road=single&speed=30&light=dark_lit
    """
    try:
        lat = float(request.GET.get("lat", 0))
        lon = float(request.GET.get("lon", 0))
        hour = int(request.GET.get("time", 12))
        day = request.GET.get("day", "Monday")
        weather = request.GET.get("weather", "fine")
        road = request.GET.get("road", "single")
        speed = int(request.GET.get("speed", 30))
        light = request.GET.get("light", "daylight")

        sample = {
            "time_period": bin_time(hour),
            "day_of_week": day,
            "weather": weather,
            "road_type": road,
            "light": light,
            "speed": bin_speed(speed),
        }

        prediction, path = predict(TRAINED_TREE, sample)

        return JsonResponse({
            "status": "success",
            "prediction": prediction,
            "decision_path": path,
            "input": sample,
            "location": {"lat": lat, "lon": lon},
        })

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=400)