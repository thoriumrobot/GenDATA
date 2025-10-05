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
import java.security.cert.X509Certificate;
    @Positive
import com.sun.org.apache.xml.internal.security.exceptions.XMLSecurityException;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.Constants;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.RFC2253Parser;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.SignatureElementProxy;
    @Positive
import org.w3c.dom.Document;
    @Positive
import org.w3c.dom.Element;

    @Positive
public class XMLX509SubjectName extends SignatureElementProxy implements XMLX509DataContent {

    @Positive
    public XMLX509SubjectName(Element element, String baseURI) throws XMLSecurityException {
    @Positive
    }

    @Positive
    public XMLX509SubjectName(Document doc, String X509SubjectNameString) {
    @Positive
    }

    @Positive
    public XMLX509SubjectName(Document doc, X509Certificate x509certificate) {
    @Positive
    }

    @Positive
    public String getSubjectName();

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
