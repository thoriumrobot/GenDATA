/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xml.internal.security.keys.storage.implementations;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.cert.Certificate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import com.sun.org.apache.xml.internal.security.keys.storage.StorageResolverException;
import com.sun.org.apache.xml.internal.security.keys.storage.StorageResolverSpi;

public class KeyStoreResolver extends StorageResolverSpi {

    public KeyStoreResolver(KeyStore keyStore) throws StorageResolverException {
    }

    public Iterator<Certificate> getIterator();

    static class KeyStoreIterator implements Iterator<Certificate> {

        public KeyStoreIterator(KeyStore keyStore) {
        }

        @Pure
        public boolean hasNext();

        @SideEffectsOnly("this")
        public Certificate next();

        public void remove();
    }
}
