/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xml.internal.security.keys.content;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.math.BigInteger;
    @Positive
import java.security.cert.X509Certificate;
    @Positive
import com.sun.org.apache.xml.internal.security.exceptions.XMLSecurityException;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.x509.XMLX509CRL;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.x509.XMLX509Certificate;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.x509.XMLX509Digest;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.x509.XMLX509IssuerSerial;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.x509.XMLX509SKI;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.content.x509.XMLX509SubjectName;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.Constants;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.SignatureElementProxy;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;

    @Positive
public class X509Data extends SignatureElementProxy implements KeyInfoContent {

    @Positive
    public X509Data(Document doc) {
    @Positive
    }

    @Positive
    public X509Data(Element element, String baseURI) throws XMLSecurityException {
    @Positive
    }

    @Positive
    public void addIssuerSerial(String X509IssuerName, BigInteger X509SerialNumber);

    @Positive
    public void addIssuerSerial(String X509IssuerName, String X509SerialNumber);

    @Positive
    public void addIssuerSerial(String X509IssuerName, int X509SerialNumber);

    @Positive
    public void add(XMLX509IssuerSerial xmlX509IssuerSerial);

    @Positive
    public void addSKI(byte[] skiBytes);

    @Positive
    public void addSKI(X509Certificate x509certificate) throws XMLSecurityException;

    @Positive
    public void add(XMLX509SKI xmlX509SKI);

    @Positive
    public void addSubjectName(String subjectName);

    @Positive
    public void addSubjectName(X509Certificate x509certificate);

    @Positive
    public void add(XMLX509SubjectName xmlX509SubjectName);

    @Positive
    public void addCertificate(X509Certificate x509certificate) throws XMLSecurityException;

    @Positive
    public void addCertificate(byte[] x509certificateBytes);

    @Positive
    public void add(XMLX509Certificate xmlX509Certificate);

    @Positive
    public void addCRL(byte[] crlBytes);

    @Positive
    public void add(XMLX509CRL xmlX509CRL);

    @Positive
    public void addDigest(X509Certificate x509certificate, String algorithmURI) throws XMLSecurityException;

    @Positive
    public void addDigest(byte[] x509CertificateDigestBytes, String algorithmURI);

    @Positive
    public void add(XMLX509Digest xmlX509Digest);

    @Positive
    public void addUnknownElement(Element element);

    @Positive
    public int lengthIssuerSerial();

    @Positive
    public int lengthSKI();

    @Positive
    public int lengthSubjectName();

    @Positive
    public int lengthCertificate();

    @Positive
    public int lengthCRL();

    @Positive
    public int lengthDigest();

    @Positive
    public int lengthUnknownElement();

    @Positive
    public XMLX509IssuerSerial itemIssuerSerial(int i) throws XMLSecurityException;

    @Positive
    public XMLX509SKI itemSKI(int i) throws XMLSecurityException;

    @Positive
    public XMLX509SubjectName itemSubjectName(int i) throws XMLSecurityException;

    @Positive
    public XMLX509Certificate itemCertificate(int i) throws XMLSecurityException;

    @Positive
    public XMLX509CRL itemCRL(int i) throws XMLSecurityException;

    @Positive
    public XMLX509Digest itemDigest(int i) throws XMLSecurityException;

    @Positive
    public Element itemUnknownElement(int i);

    @Positive
    @Pure
    @Positive
    public boolean containsIssuerSerial();

    @Positive
    @Pure
    @Positive
    public boolean containsSKI();

    @Positive
    @Pure
    @Positive
    public boolean containsSubjectName();

    @Positive
    @Pure
    @Positive
    public boolean containsCertificate();

    @Positive
    @Pure
    @Positive
    public boolean containsDigest();

    @Positive
    @Pure
    @Positive
    public boolean containsCRL();

    @Positive
    @Pure
    @Positive
    public boolean containsUnknownElement();

    @Positive
    public String getBaseLocalName();
    @Positive
}
