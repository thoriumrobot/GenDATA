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
import java.security.Key;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.SignatureException;
import java.security.spec.AlgorithmParameterSpec;
import javax.xml.crypto.MarshalException;
import javax.xml.crypto.dom.DOMCryptoContext;
import javax.xml.crypto.dsig.SignatureMethod;
import javax.xml.crypto.dsig.SignedInfo;
import javax.xml.crypto.dsig.XMLSignature;
import javax.xml.crypto.dsig.XMLSignatureException;
import javax.xml.crypto.dsig.XMLSignContext;
import javax.xml.crypto.dsig.XMLValidateContext;
import javax.xml.crypto.dsig.spec.SignatureMethodParameterSpec;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

abstract class AbstractDOMSignatureMethod extends DOMStructure implements SignatureMethod {

    abstract boolean verify(Key key, SignedInfo si, byte[] sig, XMLValidateContext context) throws InvalidKeyException, SignatureException, XMLSignatureException;

    abstract byte[] sign(Key key, SignedInfo si, XMLSignContext context) throws InvalidKeyException, XMLSignatureException;

    abstract String getJCAAlgorithm();

    abstract Type getAlgorithmType();

    @Override
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    void marshalParams(Element parent, String paramsPrefix) throws MarshalException;

    SignatureMethodParameterSpec unmarshalParams(Element paramsElem) throws MarshalException;

    void checkParams(SignatureMethodParameterSpec params) throws InvalidAlgorithmParameterException;

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    @Override
    public int hashCode();

    boolean paramsEqual(AlgorithmParameterSpec spec);
}
