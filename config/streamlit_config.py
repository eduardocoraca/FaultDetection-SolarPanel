import yaml
import streamlit as st
import streamlit_authenticator as stauth

def update_widget():
    def update(criterio_sf, criterio_tr, config_app):
        '''Updates criteria values'''
        config_app['criterios']['solda_fria'] = criterio_sf
        config_app['criterios']['trinca'] = criterio_tr
        with open('data/config.yml', 'w') as c:
            yaml.dump(config_app, c)
        st.warning('Arquivo modificado.')

    # loading data
    with open('data/config.yml', 'r') as c:
        config_app = yaml.safe_load(c)
    criterio_sf = config_app['criterios']['solda_fria']
    criterio_tr = config_app['criterios']['trinca']

    with st.expander("Atualizar critérios"):
        st.subheader("Valores atuais")
        col1, col2 = st.columns(2)
        col1.metric("Solda fria (%)", f"{config_app['criterios']['solda_fria']}")
        col2.metric("Trinca (%)", f"{config_app['criterios']['trinca']}")

        st.subheader("Novos valores")
        criterio_sf = st.slider('Critério de solda fria (>X%):', min_value=0.0, max_value=10.0, value=config_app['criterios']['solda_fria'])
        criterio_tr = st.slider('Critério de trinca (>X%):', min_value=0.0, max_value=10.0, value=config_app['criterios']['trinca'])
        st.button('Atualizar', key=None, help=None, on_click=update, args=(criterio_sf, criterio_tr, config_app))

def change_password_widget():
    with st.expander("Alterar senha"):
        try:
            if authenticator.reset_password(username, ''):
                st.success('Senha alterada')
            with open('data/config_auth.yml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)

def new_user_widget():
    with st.expander("Cadastrar novo usuário"):
        try:
            if authenticator.register_user('Register user', preauthorization=False):
                st.success('Usuário cadastrado')
                with open('data/config_auth.yml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)

# Main
with open('data/config_auth.yml') as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, status, username = authenticator.login('Login', 'main')

if status:
    update_widget()
    change_password_widget()
    new_user_widget()
    authenticator.logout('Logout')