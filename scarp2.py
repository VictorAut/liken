import pandas as pd


def id():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def address():
    return [
        "123ab, OL5 9PL, UK",
        "99 Ambleside avenue park Road, ED3 3RT, Edinburgh, United Kingdom",
        "Calle Ancho, 12, 05688, Rioja, Navarra, Espana",
        "Calle Sueco, 56, 05688, Rioja, Navarra",
        "4 Brinkworth Way, GH9 5KL, Edinburgh, United Kingdom",
        "66b Porters street, OL5 9PL, Newark, United Kingdom",
        "C. Ancho 49, 05687, Navarra",
        "Ambleside avenue Park Road ED3, UK",
        "123ab, OL5 9PL, UK",
        "123ab, OL5 9PL, UK",
        "37 Lincolnshire lane, GH9 5DF, Edinburgh, UK",
        "37 GH9, UK",
        "totally random non existant address",
    ]


def email():
    return [
        "bbab@example.com",
        "bb@example.com",
        "a@example.com",
        "hellothere@example.com",
        "b@example.com",
        "bab@example.com",
        "b@example.com",
        "hellthere@example.com",
        "hellathere@example.com",
        "irrelevant@hotmail.com",
        "yet.another.email@msn.com",
        "awesome_surfer_77@yahoo.com",
        "fictitious@never.co.uk",
    ]


def account():
    return [
        "reddit",
        "reddit",
        "facebook",
        "pinterest",
        "pinterest",
        "flickr",
        "reddit",
        "reddit",
        "facebook",
        "google",
        "flickr",
        "linkedin",
        "linkedin",
    ]


def birth_country():
    return [
        "spain",
        "spain",
        "germany",
        "japan",
        "malaysia",
        "malaysia",
        "japan",
        "germany",
        "japan",
        "malaysia",
        "germany",
        "france",
        "japan",
    ]


def marital_status():
    return [
        "married",
        "married",
        "single",
        "married",
        "single",
        "single",
        "married",
        "married",
        "single",
        "divorced",
        "married",
        "married",
        "single",
    ]


def number_children():
    return [
        1,
        0,
        2,
        0,
        0,
        0,
        1,
        0,
        3,
        1,
        1,
        1,
        0,
    ]


def property_type():
    return [
        "rental",
        "rental",
        "rental",
        "owner",
        "rental",
        "owner",
        "rental",
        "owner",
        "owner",
        "social housing",
        "rental",
        "rental",
        "rental",
    ]


def property_height():
    return [
        2.4,
        2.4,
        2.5,
        4,
        6,
        2.5,
        2.5,
        2.5,
        2.5,
        2.5,
        2.5,
        2.7,
        4,
    ]


def property_area_sq_ft():
    return [
        545,
        452,
        623,
        2077,
        3656,
        4000,
        1323,
        509,
        500,
        450,
        345,
        1045,
        1545,
    ]


def property_sea_level_elevation_m():
    return [
        5,
        6,
        5,
        305,
        1342,
        25,
        132,
        200,
        300,
        15,
        22,
        42,
        62,
    ]


def property_num_rooms():
    return [
        3,
        3,
        3,
        6,
        7,
        8,
        4,
        2,
        3,
        3,
        3,
        4,
        6,
    ]


def blocking_key():
    return [
        "key_2",
        "key_2",
        "key_2",
        "key_2",
        "key_2",
        "key_2",
        "key_1",
        "key_1",
        "key_1",
        "key_1",
        "key_1",
        "key_1",
        "key_1",
    ]


a = pd.DataFrame(
    {
        "id": id(),
        "address": address(),
        "email": email(),
        "account": account(),
        "birth_country": birth_country(),
        "marital_status": marital_status(),
        "number_children": number_children(),
        "property_type": property_type(),
        "property_height": property_height(),
        "property_area_sq_ft": property_area_sq_ft(),
        "property_sea_level_elevation_m": property_sea_level_elevation_m(),
        "property_num_rooms": property_num_rooms(),
    }
)
a
