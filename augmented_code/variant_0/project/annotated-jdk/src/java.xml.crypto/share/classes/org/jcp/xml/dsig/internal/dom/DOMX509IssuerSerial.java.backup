/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package org.jcp.xml.dsig.internal.dom;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import javax.xml.crypto.MarshalException;
import javax.xml.crypto.dom.DOMCryptoContext;
import javax.xml.crypto.dsig.XMLSignature;
import javax.xml.crypto.dsig.keyinfo.X509IssuerSerial;
import java.math.BigInteger;
import javax.security.auth.x500.X500Principal;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

public final class DOMX509IssuerSerial extends DOMStructure implements X509IssuerSerial {

    public DOMX509IssuerSerial(String issuerName, BigInteger serialNumber) {
    }

    public DOMX509IssuerSerial(Element isElem) throws MarshalException {
    }

    public String getIssuerName();

    public BigInteger getSerialNumber();

    @Override
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Override
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();
}
