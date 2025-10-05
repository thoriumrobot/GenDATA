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
import java.security.cert.X509Certificate;
import java.util.Arrays;
import com.sun.org.apache.xml.internal.security.exceptions.XMLSecurityException;
import com.sun.org.apache.xml.internal.security.utils.Constants;
import com.sun.org.apache.xml.internal.security.utils.SignatureElementProxy;
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

public class XMLX509SKI extends SignatureElementProxy implements XMLX509DataContent {

    public static final String SKI_OID;

    public XMLX509SKI(Document doc, byte[] skiBytes) {
    }

    public XMLX509SKI(Document doc, X509Certificate x509certificate) throws XMLSecurityException {
    }

    public XMLX509SKI(Element element, String baseURI) throws XMLSecurityException {
    }

    public byte[] getSKIBytes() throws XMLSecurityException;

    public static byte[] getSKIBytesFromCert(X509Certificate cert) throws XMLSecurityException;

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public String getBaseLocalName();
}
