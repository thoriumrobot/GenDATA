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
import javax.xml.crypto.*;
import javax.xml.crypto.dom.DOMCryptoContext;
import javax.xml.crypto.dsig.*;
import javax.xml.crypto.dsig.spec.RSAPSSParameterSpec;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;
import java.security.Provider;
import java.security.spec.AlgorithmParameterSpec;
import java.security.spec.MGF1ParameterSpec;
import java.security.spec.PSSParameterSpec;
import java.util.*;
import com.sun.org.apache.xml.internal.security.algorithms.implementations.SignatureBaseRSA;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import com.sun.org.apache.xml.internal.security.utils.UnsyncBufferedOutputStream;
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;

public final class DOMSignedInfo extends DOMStructure implements SignedInfo {

    public DOMSignedInfo(CanonicalizationMethod cm, SignatureMethod sm, List<? extends Reference> references) {
    }

    public DOMSignedInfo(CanonicalizationMethod cm, SignatureMethod sm, List<? extends Reference> references, String id) {
    }

    public DOMSignedInfo(Element siElem, XMLCryptoContext context, Provider provider) throws MarshalException {
    }

    public CanonicalizationMethod getCanonicalizationMethod();

    public SignatureMethod getSignatureMethod();

    public String getId();

    public List<Reference> getReferences();

    public InputStream getCanonicalizedData();

    public void canonicalize(XMLCryptoContext context, ByteArrayOutputStream bos) throws XMLSignatureException;

    @Override
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @SuppressWarnings("unchecked")
    public static List<Reference> getSignedInfoReferences(SignedInfo si);

    @Override
    public int hashCode();
}
