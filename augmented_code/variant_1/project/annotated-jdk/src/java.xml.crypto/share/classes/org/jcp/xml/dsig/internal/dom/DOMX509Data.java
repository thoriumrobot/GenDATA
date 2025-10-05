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
package org.jcp.xml.dsig.internal.dom;

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
import java.security.cert.CRLException;
    @Positive
import java.security.cert.CertificateEncodingException;
    @Positive
import java.security.cert.CertificateException;
    @Positive
import java.security.cert.CertificateFactory;
    @Positive
import java.security.cert.X509CRL;
    @Positive
import java.security.cert.X509Certificate;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.List;
    @Positive
import javax.security.auth.x500.X500Principal;
    @Positive
import javax.xml.crypto.MarshalException;
    @Positive
import javax.xml.crypto.XMLStructure;
    @Positive
import javax.xml.crypto.dom.DOMCryptoContext;
    @Positive
import javax.xml.crypto.dsig.XMLSignature;
    @Positive
import javax.xml.crypto.dsig.keyinfo.X509Data;
    @Positive
import javax.xml.crypto.dsig.keyinfo.X509IssuerSerial;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;

    @Positive
public final class DOMX509Data extends DOMStructure implements X509Data {

    @Positive
    public DOMX509Data(List<?> content) {
    @Positive
    }

    @Positive
    public DOMX509Data(Element xdElem) throws MarshalException {
    @Positive
    }

    @Positive
    public List<Object> getContent();

    @Positive
    @Override
    @Positive
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    @Override
    @Positive
    public int hashCode();
    @Positive
}
