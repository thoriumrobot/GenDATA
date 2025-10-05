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
import java.io.IOException;
import java.math.BigInteger;
import java.security.KeyException;
import java.security.KeyFactory;
import java.security.NoSuchAlgorithmException;
import java.security.PublicKey;
import java.security.interfaces.DSAParams;
import java.security.interfaces.DSAPublicKey;
import java.security.interfaces.ECPublicKey;
import java.security.interfaces.RSAPublicKey;
import java.security.spec.DSAPublicKeySpec;
import java.security.spec.ECField;
import java.security.spec.ECFieldFp;
import java.security.spec.ECParameterSpec;
import java.security.spec.ECPoint;
import java.security.spec.ECPublicKeySpec;
import java.security.spec.EllipticCurve;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.KeySpec;
import java.security.spec.RSAPublicKeySpec;
import java.util.Arrays;
import javax.xml.crypto.MarshalException;
import javax.xml.crypto.dom.DOMCryptoContext;
import javax.xml.crypto.dsig.XMLSignature;
import javax.xml.crypto.dsig.keyinfo.KeyValue;
import com.sun.org.apache.xml.internal.security.utils.XMLUtils;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

public abstract class DOMKeyValue<K extends PublicKey> extends DOMStructure implements KeyValue {

    public DOMKeyValue(K key) throws KeyException {
    }

    public DOMKeyValue(Element kvtElem) throws MarshalException {
    }

    static KeyValue unmarshal(Element kvElem) throws MarshalException;

    public PublicKey getPublicKey() throws KeyException;

    @Override
    public void marshal(Node parent, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    abstract void marshalPublicKey(Node parent, Document doc, String dsPrefix, DOMCryptoContext context) throws MarshalException;

    abstract K unmarshalKeyValue(Element kvtElem) throws MarshalException;

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public static BigInteger decode(Element elem) throws MarshalException;

    @Override
    public int hashCode();

    static final class RSA extends DOMKeyValue<RSAPublicKey> {

        void marshalPublicKey(Node parent, Document doc, String dsPrefix, DOMCryptoContext context) throws MarshalException;

        @Override
        RSAPublicKey unmarshalKeyValue(Element kvtElem) throws MarshalException;
    }

    static final class DSA extends DOMKeyValue<DSAPublicKey> {

        @Override
        void marshalPublicKey(Node parent, Document doc, String dsPrefix, DOMCryptoContext context) throws MarshalException;

        @Override
        DSAPublicKey unmarshalKeyValue(Element kvtElem) throws MarshalException;
    }

    static final class EC extends DOMKeyValue<ECPublicKey> {

        @Override
        void marshalPublicKey(Node parent, Document doc, String dsPrefix, DOMCryptoContext context) throws MarshalException;

        @Override
        ECPublicKey unmarshalKeyValue(Element kvtElem) throws MarshalException;

        static final class Curve extends ECParameterSpec {
        }
    }

    static final class Unknown extends DOMKeyValue<PublicKey> {

        @Override
        PublicKey unmarshalKeyValue(Element kvElem) throws MarshalException;

        @Override
        void marshalPublicKey(Node parent, Document doc, String dsPrefix, DOMCryptoContext context) throws MarshalException;
    }
}
