/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xml.internal.security.keys.storage.implementations;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import java.security.cert.Certificate;
import java.security.cert.X509Certificate;
import java.util.Iterator;
import java.util.NoSuchElementException;
import com.sun.org.apache.xml.internal.security.keys.storage.StorageResolverSpi;

public class SingleCertificateResolver extends StorageResolverSpi {

    public SingleCertificateResolver(X509Certificate x509cert) {
    }

    public Iterator<Certificate> getIterator();

    static class InternalIterator implements Iterator<Certificate> {

        public InternalIterator(X509Certificate x509cert) {
        }

        @Pure
        public boolean hasNext();

        @SideEffectsOnly("this")
        public Certificate next();

        public void remove();
    }
}
