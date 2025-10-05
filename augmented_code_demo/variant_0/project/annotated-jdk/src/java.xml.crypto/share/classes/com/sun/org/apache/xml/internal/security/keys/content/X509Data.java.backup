/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xml.internal.security.keys.content;

import org.checkerframework.dataflow.qual.Pure;
import java.math.BigInteger;
import java.security.cert.X509Certificate;
import com.sun.org.apache.xml.internal.security.exceptions.XMLSecurityException;
import com.sun.org.apache.xml.internal.security.keys.content.x509.XMLX509CRL;
import com.sun.org.apache.xml.internal.security.keys.content.x509.XMLX509Certificate;
import com.sun.org.apache.xml.internal.security.keys.content.x509.XMLX509Digest;
import com.sun.org.apache.xml.internal.security.keys.content.x509.XMLX509IssuerSerial;
import com.sun.org.apache.xml.internal.security.keys.content.x509.XMLX509SKI;
import com.sun.org.apache.xml.internal.security.keys.content.x509.XMLX509SubjectName;
import com.sun.org.apache.xml.internal.security.utils.Constants;
import com.sun.org.apache.xml.internal.security.utils.SignatureElementProxy;
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

public class X509Data extends SignatureElementProxy implements KeyInfoContent {

    public X509Data(Document doc) {
    }

    public X509Data(Element element, String baseURI) throws XMLSecurityException {
    }

    public void addIssuerSerial(String X509IssuerName, BigInteger X509SerialNumber);

    public void addIssuerSerial(String X509IssuerName, String X509SerialNumber);

    public void addIssuerSerial(String X509IssuerName, int X509SerialNumber);

    public void add(XMLX509IssuerSerial xmlX509IssuerSerial);

    public void addSKI(byte[] skiBytes);

    public void addSKI(X509Certificate x509certificate) throws XMLSecurityException;

    public void add(XMLX509SKI xmlX509SKI);

    public void addSubjectName(String subjectName);

    public void addSubjectName(X509Certificate x509certificate);

    public void add(XMLX509SubjectName xmlX509SubjectName);

    public void addCertificate(X509Certificate x509certificate) throws XMLSecurityException;

    public void addCertificate(byte[] x509certificateBytes);

    public void add(XMLX509Certificate xmlX509Certificate);

    public void addCRL(byte[] crlBytes);

    public void add(XMLX509CRL xmlX509CRL);

    public void addDigest(X509Certificate x509certificate, String algorithmURI) throws XMLSecurityException;

    public void addDigest(byte[] x509CertificateDigestBytes, String algorithmURI);

    public void add(XMLX509Digest xmlX509Digest);

    public void addUnknownElement(Element element);

    public int lengthIssuerSerial();

    public int lengthSKI();

    public int lengthSubjectName();

    public int lengthCertificate();

    public int lengthCRL();

    public int lengthDigest();

    public int lengthUnknownElement();

    public XMLX509IssuerSerial itemIssuerSerial(int i) throws XMLSecurityException;

    public XMLX509SKI itemSKI(int i) throws XMLSecurityException;

    public XMLX509SubjectName itemSubjectName(int i) throws XMLSecurityException;

    public XMLX509Certificate itemCertificate(int i) throws XMLSecurityException;

    public XMLX509CRL itemCRL(int i) throws XMLSecurityException;

    public XMLX509Digest itemDigest(int i) throws XMLSecurityException;

    public Element itemUnknownElement(int i);

    @Pure
    public boolean containsIssuerSerial();

    @Pure
    public boolean containsSKI();

    @Pure
    public boolean containsSubjectName();

    @Pure
    public boolean containsCertificate();

    @Pure
    public boolean containsDigest();

    @Pure
    public boolean containsCRL();

    @Pure
    public boolean containsUnknownElement();

    public String getBaseLocalName();
}
