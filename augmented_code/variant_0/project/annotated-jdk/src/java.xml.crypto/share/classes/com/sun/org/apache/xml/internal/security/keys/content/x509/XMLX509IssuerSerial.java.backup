/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xml.internal.security.keys.content.x509;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.math.BigInteger;
import java.security.cert.X509Certificate;
import com.sun.org.apache.xml.internal.security.exceptions.XMLSecurityException;
import com.sun.org.apache.xml.internal.security.utils.Constants;
import com.sun.org.apache.xml.internal.security.utils.RFC2253Parser;
import com.sun.org.apache.xml.internal.security.utils.SignatureElementProxy;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

public class XMLX509IssuerSerial extends SignatureElementProxy implements XMLX509DataContent {

    public XMLX509IssuerSerial(Element element, String baseURI) throws XMLSecurityException {
    }

    public XMLX509IssuerSerial(Document doc, String x509IssuerName, BigInteger x509SerialNumber) {
    }

    public XMLX509IssuerSerial(Document doc, String x509IssuerName, String x509SerialNumber) {
    }

    public XMLX509IssuerSerial(Document doc, String x509IssuerName, int x509SerialNumber) {
    }

    public XMLX509IssuerSerial(Document doc, X509Certificate x509certificate) {
    }

    public BigInteger getSerialNumber();

    public int getSerialNumberInteger();

    public String getIssuerName();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public String getBaseLocalName();
}
