/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xml.internal.security.keys.keyresolver;

import java.lang.reflect.InvocationTargetException;
import java.security.PublicKey;
import java.security.cert.X509Certificate;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.DEREncodedKeyValueResolver;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.DSAKeyValueResolver;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.ECKeyValueResolver;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.KeyInfoReferenceResolver;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.RSAKeyValueResolver;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.RetrievalMethodResolver;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.X509CertificateResolver;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.X509DigestResolver;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.X509IssuerSerialResolver;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.X509SKIResolver;
import com.sun.org.apache.xml.internal.security.keys.keyresolver.implementations.X509SubjectNameResolver;
import com.sun.org.apache.xml.internal.security.keys.storage.StorageResolver;
import com.sun.org.apache.xml.internal.security.utils.JavaUtils;

public class KeyResolver {

    public static int length();

    public static final X509Certificate getX509Certificate(Element element, String baseURI, StorageResolver storage, boolean secureValidation) throws KeyResolverException;

    public static final PublicKey getPublicKey(Element element, String baseURI, StorageResolver storage, boolean secureValidation) throws KeyResolverException;

    public static void register(String className) throws ClassNotFoundException, IllegalAccessException, InstantiationException, InvocationTargetException;

    public static void registerAtStart(String className);

    public static void register(KeyResolverSpi keyResolverSpi, boolean start);

    public static void registerClassNames(List<String> classNames) throws ClassNotFoundException, IllegalAccessException, InstantiationException, InvocationTargetException;

    public static void registerDefaultResolvers();

    static class ResolverIterator implements Iterator<KeyResolverSpi> {

        public ResolverIterator(List<KeyResolverSpi> list) {
        }

        @Pure
        public boolean hasNext();

        @SideEffectsOnly("this")
        public KeyResolverSpi next();

        public void remove();
    }

    public static Iterator<KeyResolverSpi> iterator();
}
