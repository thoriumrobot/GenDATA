/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xml.internal.security.keys.content.x509;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.ByteArrayInputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.security.PublicKey;
    @Positive
import java.security.cert.CertificateException;
    @Positive
import java.security.cert.CertificateFactory;
    @Positive
import java.security.cert.X509Certificate;
    @Positive
import java.util.Arrays;
    @Positive
import com.sun.org.apache.xml.internal.security.exceptions.XMLSecurityException;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.Constants;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.SignatureElementProxy;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;

    @Positive
public class XMLX509Certificate extends SignatureElementProxy implements XMLX509DataContent {

    @Positive
    public static final String JCA_CERT_ID;

    @Positive
    public XMLX509Certificate(Element element, String baseURI) throws XMLSecurityException {
    @Positive
    }

    @Positive
    public XMLX509Certificate(Document doc, byte[] certificateBytes) {
    @Positive
    }

    @Positive
    public XMLX509Certificate(Document doc, X509Certificate x509certificate) throws XMLSecurityException {
    @Positive
    }

    @Positive
    public byte[] getCertificateBytes() throws XMLSecurityException;

    @Positive
    public X509Certificate getX509Certificate() throws XMLSecurityException;

    @Positive
    public PublicKey getPublicKey() throws XMLSecurityException, IOException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public String getBaseLocalName();
    @Positive
}

// CFWR semantic augmentation - variant 1
