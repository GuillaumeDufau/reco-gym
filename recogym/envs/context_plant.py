class Context:
    """
    Basic context element with only simple weather.
    Either sun or rain at random
    """
    def time(self):
        raise NotImplemented

    def user(self):
        raise NotImplemented

class Context_v1(Context):
    def __init__(self, current_time, current_plant_id, maturity, water_level):
        #(self, current_time, rain_change):
        super(Context_v1, self).__init__()
        self.current_time = current_time
        self.current_plant_id = current_plant_id
        #self.rain_chance = rain_chance
        self._maturity = maturity
        self._water_level = water_level

    def time(self):
        return self.current_time

    def plant(self):
        return self.current_plant_id

    def water_level(self):
        return self._water_level

    def maturity(self):
        return self._maturity

    

class DefaultContext(Context):

    def __init__(self, current_time, current_user_id):
        super(DefaultContext, self).__init__()
        self.current_time = current_time
        self.current_user_id = current_user_id

    def time(self):
        return self.current_time

    def user(self):
        return self.current_user_id
