import React from "react";
import { Redirect } from "react-router-dom";
import Auth_2 from "../components/Auth_2";

const Auth_Page_2 = ({ authType, isAuthenticated }) => {
  if (isAuthenticated) {
    return <Redirect to="/student" />;
  }

  return (
    <div>
      {/* eslint-disable-next-line react/jsx-pascal-case */}
      <Auth_2 authType={authType} />
    </div>
  );
};

export default Auth_Page_2;
