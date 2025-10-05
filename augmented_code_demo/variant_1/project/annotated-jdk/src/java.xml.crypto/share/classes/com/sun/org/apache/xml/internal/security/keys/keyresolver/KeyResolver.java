/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xml.internal.security.keys.keyresolver;

    @Positive
import java.lang.reflect.InvocationTargetException;
    @Positive
import java.security.PublicKey;
    @Positive
import java.security.cert.X509Certificate;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.concurrent.CopyOnWriteArrayList;
    @Positive
import java.util.concurrent.atomic.AtomicBoolean;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.w3c.dom.Element;
    @Positive
import org.w3c.dom.Node;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.DEREncodedKeyValueResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.DSAKeyValueResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.ECKeyValueResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.KeyInfoReferenceResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.RSAKeyValueResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.RetrievalMethodResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.X509CertificateResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.X509DigestResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.X509IssuerSerialResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.X509SKIResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.X509SubjectNameResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.keys.storage.StorageResolver;
    @Positive
import com.sun.org.apache.xml.internal.security.utils.JavaUtils;

    @Positive
public class KeyResolver {

    @Positive
    public static int length();

    @Positive
    public static final X509Certificate getX509Certificate(Element element, String baseURI, StorageResolver storage, boolean secureValidation) throws KeyResolverException;

    @Positive
    public static final PublicKey getPublicKey(Element element, String baseURI, StorageResolver storage, boolean secureValidation) throws KeyResolverException;

    @Positive
    public static void register(String className) throws ClassNotFoundException, IllegalAccessException, InstantiationException, InvocationTargetException;

    @Positive
    public static void registerAtStart(String className);

    @Positive
    public static void register(KeyResolverSpi keyResolverSpi, boolean start);

    @Positive
    public static void registerClassNames(List<String> classNames) throws ClassNotFoundException, IllegalAccessException, InstantiationException, InvocationTargetException;

    @Positive
    public static void registerDefaultResolvers();

    @Positive
    static class ResolverIterator implements Iterator<KeyResolverSpi> {

    @Positive
        public ResolverIterator(List<KeyResolverSpi> list) {
    @Positive
        }

    @Positive
        @Pure
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public KeyResolverSpi next();

    @Positive
        public void remove();
    @Positive
    }

    @Positive
    public static Iterator<KeyResolverSpi> iterator();
    @Positive
}

// CFWR semantic augmentation - variant 1
