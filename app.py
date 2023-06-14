
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import base64


# step 1 load pickle files

with open('catboost.pkl', 'rb') as file:
    model_catboost = pickle.load(file)

with open('avg_mpg_transformation.pkl', 'rb') as file:
    avg_mpg_transformation = pickle.load(file)
    
with open('body_type_of_the_car_Encoding.pkl', 'rb') as file:
    body_type_of_the_car_Encoding = pickle.load(file)

with open('driver_rating_transformation.pkl', 'rb') as file:
    driver_rating_transformation = pickle.load(file)

with open('driver_reviews_num_transformation.pkl', 'rb') as file:
    driver_reviews_num_transformation = pickle.load(file)

with open('drivetrain_Encoding.pkl', 'rb') as file:
    drivetrain_Encoding = pickle.load(file)

with open('exterior_color_Encoding.pkl', 'rb') as file:
    exterior_color_Encoding = pickle.load(file)
    
with open('fuel_type_Encoding.pkl', 'rb') as file:
    fuel_type_Encoding = pickle.load(file)

with open('interior_color_Encoding.pkl', 'rb') as file:
    interior_color_Encoding = pickle.load(file)
    
with open('manufacturer_Encoding.pkl', 'rb') as file:
    manufacturer_Encoding = pickle.load(file)
    
with open('mileage_transformation.pkl', 'rb') as file:
    mileage_transformation = pickle.load(file)
    
with open('seller_rating_transformation.pkl', 'rb') as file:
    seller_rating_transformation = pickle.load(file)
    
with open('transmission_Encoding.pkl', 'rb') as file:
    transmission_Encoding = pickle.load(file)
    
with open('year_Encoding.pkl', 'rb') as file:
    year_Encoding = pickle.load(file)

# step 2 : get input from user

with open("carimage.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
st.markdown(
f"""
<style>
.stApp {{
    background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
    background-size: cover
}}
</style>
""",
unsafe_allow_html=True
)

st.title("Used cars price")
# st.markdown("Will this customer honour the booking? ")

# 1
manufacturer = st.selectbox('Manufacturer', ('ACURA', 'AUDI', 'BMW', 'BUICK', 'CADILLAC', 'CHEVROLET', 
                                             'CHRYSLER', 'DODGE', 'FORD', 'GMC', 'HONDA', 'HYUNDAI', 'INFINITI',
                                             'JAGUAR', 'JEEP', 'KIA', 'LAND ROVER', 'LEXUS', 'LINCOLN', 'MAZDA',
                                             'MERCEDES-BENZ', 'MITSUBISHI', 'NISSAN', 'PORSCHE', 'RAM', 'SUBARU',
                                             'TESLA', 'TOYOTA', 'VOLKSWAGEN', 'VOLVO'))

# 2
body_type_of_the_car = st.selectbox('Body Type', ('COUPE', 'SUV', 'SEDAN', 'CONVERTIBLE', 'TRUCK', 'ROADSTER', 
                                                  'HATCHBACK', 'MINIVAN', 'VAN', 'WAGON', 'ELECTRICVEHICLE'))
# 3
year = st.slider('Year of Manufactured', 1915, 2023)

# 4
mileage = st.slider('Mileage car driven', 10,1119067)

# 5
drivetrain = st.selectbox('Drive train', ('RWD', 'AWD', 'FWD', '4WD', 'FWD LSD', 'AWD LLSD', 'FWD LLSD', 
                                          'RWD LSD', 'AWD LD', 'FWD LD'))

# 6
fuel_type = st.selectbox('Fuel Type', ('Gasoline', 'Hybrid', 'Premium Gasoline', 'Electric', 
                                             'Gasoline/Mild Electric Hybrid', 'Diesel', 'E85 Flex Fuel',
                                             'Flexible Fuel', 'Plug-In Hybrid', 'Compressed Natural Gas', 
                                             ' Hybrid', ' Gasoline-Electric Hybrid', ' Hydrogen', 'Plug-In Electric/Gas'))

# 7
transmission = st.selectbox('Fuel Type', ('MANUAL', 'AUTOMANUAL', 'AUTOMATIC', 'DCT', 'CVT', 'ECVT', 'SEMI AUTOMATIC'))

# 8
avg_mpg = st.slider('Average MPG',6,142)

# 9
exterior_color = st.selectbox('Exterior color', ('RED', 'WHITE', 'GRAY', 'BLUE', 'SILVER', 'YELLOW',
                                                 'BLACK', 'GOLD', 'MAROON', 'BEIGE', 'BROWN', 'GREEN', 'PURPLE',
                                                 'ORANGE', 'PINK', 'VIOLET', 'GLACIER', 'PEARL', 'METALLIC'))


# 10
interior_color = st.selectbox('Interior color', ('BLACK', 'RED', 'SILVER', 'BEIGE', 'GRAY', 'BROWN', 'BLUE', 
                                                 'WHITE', 'YELLOW', 'GREEN', 'GOLD'))

# 11
accidents_or_damage = st.selectbox('Accident or damage', ('Yes' ,'No'))

# 12
one_owner = st.selectbox('One Owner', ('Yes' ,'No'))

# 13
personal_use_only = st.selectbox('Personal use',('Yes' ,'No'))

# 14
seller_rating = st.slider('Seller Rating', 0,5)

# 15
driver_rating = st.slider('Driver Rating', 0,5)

# 16
driver_reviews_num = st.slider('Number of Driver reviews', 0 , 1025)


# step 3 -  Data preprocessing

# encoding

bodytype_Encoding = body_type_of_the_car_Encoding.transform(pd.DataFrame({'body_type_of_the_car': [body_type_of_the_car]}))

dt_Encoding = drivetrain_Encoding.transform(pd.DataFrame({'drivetrain': [drivetrain]}))

excolor_Encoding = exterior_color_Encoding.transform(pd.DataFrame({'exterior_color': [exterior_color]}))

fuel_Encoding = fuel_type_Encoding.transform(pd.DataFrame({'fuel_type': [fuel_type]}))

intcolor_Encoding = interior_color_Encoding.transform(pd.DataFrame({'interior_color': [interior_color]}))

manu_Encoding = manufacturer_Encoding.transform(pd.DataFrame({'manufacturer':[manufacturer]}))

trans_Encoding = transmission_Encoding.transform(pd.DataFrame({'transmission':[transmission]}))

yr_Encoding = year_Encoding.transform(pd.DataFrame({'year': [year]}))

# # transformation

avgmpg_transformation = avg_mpg_transformation.transform(pd.DataFrame([avg_mpg]))
                                      
driverrating_transformation = driver_rating_transformation.transform(pd.DataFrame([driver_rating]))
                                      
driverreviews_transformation = driver_reviews_num_transformation.transform(pd.DataFrame([driver_reviews_num]))
                                      
mile_transformation = mileage_transformation.transform(pd.DataFrame([mileage]))
                                      
sellerrating_transformation = seller_rating_transformation.transform(pd.DataFrame([seller_rating]))


data  = {'manufacturer':manu_Encoding.iloc[0,0],
         'body_type_of_the_car': bodytype_Encoding.iloc[0,0],
         'year': yr_Encoding.iloc[0,0],
         'mileage': mile_transformation[0],
         'drivetrain': dt_Encoding.iloc[0,0],
         'transmission':trans_Encoding.iloc[0,0],
         'fuel_type': fuel_Encoding.iloc[0,0],
         'avg_mpg':avgmpg_transformation[0],
         'exterior_color': excolor_Encoding.iloc[0,0],
         'interior_color': intcolor_Encoding.iloc[0,0],
         'accidents_or_damage':[1 if accidents_or_damage == 'Yes' else 0],
         'one_owner': [1 if one_owner == 'Yes' else 0],
         'personal_use_only': [1 if personal_use_only == 'Yes' else 0],
         'seller_rating':sellerrating_transformation[0],
         'driver_rating':driverrating_transformation[0],
         'driver_reviews_num':driverreviews_transformation[0]
        }

predictions = model_catboost.predict(pd.DataFrame(data))

if st.button('Car Price'):
    st.subheader(round(predictions[0], 2))
