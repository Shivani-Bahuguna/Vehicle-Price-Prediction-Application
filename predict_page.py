import streamlit as st
import numpy as np
import pickle


#loading the ml model
def load_model():
    with open('saved_steps_of_vehicle_price_prediction.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()


#encoding the categorical variables
linearregressor = data["model"]
le_Make = data["le_Make"]
le_Fuel_Type = data["le_Fuel_Type"]
le_Transmission = data["le_Transmission"]
le_Owner = data["le_Owner"]
le_Drivetrain = data["le_Drivetrain"]

#function to run on page load
def show_predict_page():
    st.title("VEHICLE PRICE PREDICTION")
    st.write("""### We need some information for prediction""")

    Make = (
        'Honda', 'Maruti Suzuki', 'Hyundai', 'Toyota', 'Mercedes-Benz',
       'BMW', 'Skoda', 'Nissan', 'Renault', 'Tata', 'Volkswagen', 'Ford',
       'Audi', 'Mahindra', 'MG', 'Jeep', 'Porsche', 'Kia', 'Land Rover',
       'Volvo', 'Maserati', 'Jaguar', 'Isuzu', 'Fiat', 'MINI', 'Ferrari',
       'Mitsubishi', 'Datsun', 'Lamborghini', 'Chevrolet', 'Ssangyong',
       'Rolls-Royce', 'Lexus'
    )

    Fuel_Type = (
        'Petrol', 'Diesel', 'CNG', 'LPG', 'Electric', 'CNG + CNG',
       'Hybrid', 'Petrol + CNG', 'Petrol + LPG'
    )

    Transmission = (
        'Manual', 'Automatic'
    )

    Owner = (
        'First', 'Second', 'Third', 'Fourth', 'UnRegistered Car',
       '4 or More'
    )

    Drivetrain = (
        'FWD', 'RWD', 'AWD'
    )


    Make = st.selectbox("Make", Make)
    Fuel_Type = st.selectbox("Fuel_Type",Fuel_Type)
    Transmission = st.selectbox("Transmission",Transmission)
    Owner = st.selectbox("Owner",Owner)
    Drivetrain = st.selectbox("Drivetrain",Drivetrain)


    Year = st.number_input("Enter the model Year",min_value=2009,max_value=2022,step=1)

    Engine = st.number_input("Enter your Engine capacity in cc")
    Max_Power = st.number_input("Enter the Max Power of engine in bhp")
    Seating_Capacity = st.number_input("Enter the seating capacity")
    Fuel_Tank_Capacity = st.number_input("Enter the fuel tank capacity")


    ok = st.button("PREDICT Price")
    #if ok is true, which means the button is clicked, prediction result is to be shown
    if ok:
        XINPUT = np.array([[Make, Year, Fuel_Type, Transmission, Owner, Engine, Max_Power, Drivetrain,
                            Seating_Capacity, Fuel_Tank_Capacity]])
        XINPUT[:, 0] = le_Make.transform(XINPUT[:, 0])
        XINPUT[:, 2] = le_Fuel_Type.transform(XINPUT[:, 2])
        XINPUT[:, 3] = le_Transmission.transform(XINPUT[:, 3])
        XINPUT[:, 4] = le_Owner.transform(XINPUT[:, 4])
        XINPUT[:, 7] = le_Drivetrain.transform(XINPUT[:, 7])

        XINPUT = XINPUT.astype('float')

        price = linearregressor.predict(XINPUT)
        st.subheader("The predicted price is {}".format(price[0]))


