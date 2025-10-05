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
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.PublicKey;
import java.security.cert.CertificateException;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.util.Arrays;
import com.sun.org.apache.xml.internal.security.exceptions.XMLSecurityException;
import com.sun.org.apache.xml.internal.security.utils.Constants;
import com.sun.org.apache.xml.internal.security.utils.SignatureElementProxy;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

public class XMLX509Certificate extends SignatureElementProxy implements XMLX509DataContent {

    public static final String JCA_CERT_ID;

    public XMLX509Certificate(Element element, String baseURI) throws XMLSecurityException {
    }

    public XMLX509Certificate(Document doc, byte[] certificateBytes) {
    }

    public XMLX509Certificate(Document doc, X509Certificate x509certificate) throws XMLSecurityException {
    }

    public byte[] getCertificateBytes() throws XMLSecurityException;

    public X509Certificate getX509Certificate() throws XMLSecurityException;

    public PublicKey getPublicKey() throws XMLSecurityException, IOException;

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public String getBaseLocalName();
}
