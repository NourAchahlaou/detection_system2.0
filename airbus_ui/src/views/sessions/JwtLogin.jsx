import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Card, Grid, TextField, Box, styled } from "@mui/material";
import { LoadingButton } from "@mui/lab";
import { Formik } from "formik";
import * as Yup from "yup";

import useAuth from "app/hooks/useAuth";

// STYLED COMPONENTS
const FlexBox = styled(Box)(() => ({
  display: "flex"
}));

const ContentBox = styled("div")(() => ({
  height: "100%",
  padding: "32px",
  position: "relative",
  background: "rgba(0, 0, 0, 0.01)",
  marginTop: "5rem"
}));

const StyledRoot = styled("div")(() => ({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  backgroundColor: "#1A2038",
  minHeight: "100vh",
  "& .card": {
    maxWidth: 800,
    minHeight: 400,
    margin: "1rem",
    display: "flex",
    borderRadius: 12,
    alignItems: "center"
  },
  ".img-wrapper": {
    height: "100%",
    minWidth: 320,
    display: "flex",
    padding: "2rem",
    alignItems: "center",
    justifyContent: "center"
  }
}));

const initialValues = {
  username: "",
  password: ""
};

const validationSchema = Yup.object().shape({
  username: Yup.string().required("Username is required!"),
  password: Yup.string()
    .min(4, "Password must be at least 4 characters long")
    .required("Password is required!")
});

export default function JwtLogin() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const { token, error, loading: authLoading, login } = useAuth(); // Destructure state from useAuth

  const handleFormSubmit = async (values) => {
    setLoading(true);
    try {
      const data = await login(values.username, values.password);
      console.log("Login response data:", data);
  
      const { access_token } = data;
      if (access_token) {
        console.log("Access token received:", access_token);
        localStorage.setItem("token", access_token); // Ensure the key matches
        navigate("/profile&Settings/profile");
      } else {
        console.error("Access token is missing in the response");
      }
    } catch (e) {
      console.error("Login failed:", e);
    } finally {
      setLoading(false);
    }
  };
  

  // Effect to track changes in authState
  useEffect(() => {
    console.log("Auth state changed:", { token, error, authLoading });
  }, [token, error, authLoading]);

  return (
    <StyledRoot>
      <Card className="card">
        <Grid container>
          <Grid item sm={6} xs={12}>
            <div className="img-wrapper">
              <img
                src="/assets/images/illustrations/Flying_around_the_world-amico.svg"
                width="100%"
                alt=""
              />
            </div>
          </Grid>

          <Grid item sm={6} xs={12}>
            <ContentBox>
              <Formik
                onSubmit={handleFormSubmit}
                initialValues={initialValues}
                validationSchema={validationSchema}
              >
                {({
                  values,
                  errors,
                  touched,
                  handleChange,
                  handleBlur,
                  handleSubmit,
                }) => (
                  <form onSubmit={handleSubmit}>
                    <TextField
                      fullWidth
                      size="small"
                      name="username"
                      label="Username"
                      variant="outlined"
                      onBlur={handleBlur}
                      value={values.username}
                      onChange={handleChange}
                      helperText={touched.username && errors.username}
                      error={Boolean(errors.username && touched.username)}
                      sx={{ mb: 3 }}
                    />

                    <TextField
                      fullWidth
                      size="small"
                      name="password"
                      type="password"
                      label="Password"
                      variant="outlined"
                      onBlur={handleBlur}
                      value={values.password}
                      onChange={handleChange}
                      helperText={touched.password && errors.password}
                      error={Boolean(errors.password && touched.password)}
                      sx={{ mb: 1.5 }}
                    />

                    <LoadingButton
                      type="submit"
                      color="primary"
                      loading={loading}
                      variant="contained"
                      sx={{ my: 2 }}
                    >
                      Login
                    </LoadingButton>
                  </form>
                )}
              </Formik>
            </ContentBox>
          </Grid>
        </Grid>
      </Card>
    </StyledRoot>
  );
}
