/*
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.security;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.*;
import java.util.*;
import java.security.KeyStore.*;
import java.security.cert.Certificate;
import java.security.cert.CertificateException;
import javax.crypto.SecretKey;
import javax.security.auth.callback.*;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class KeyStoreSpi {

    public KeyStoreSpi() {
    }

    public abstract Key engineGetKey(String alias, char[] password) throws NoSuchAlgorithmException, UnrecoverableKeyException;

    public abstract Certificate[] engineGetCertificateChain(String alias);

    public abstract Certificate engineGetCertificate(String alias);

    public abstract Date engineGetCreationDate(String alias);

    public abstract void engineSetKeyEntry(String alias, Key key, char[] password, Certificate[] chain) throws KeyStoreException;

    public abstract void engineSetKeyEntry(String alias, byte[] key, Certificate[] chain) throws KeyStoreException;

    public abstract void engineSetCertificateEntry(String alias, Certificate cert) throws KeyStoreException;

    public abstract void engineDeleteEntry(String alias) throws KeyStoreException;

    public abstract Enumeration<String> engineAliases();

    public abstract boolean engineContainsAlias(String alias);

    public abstract int engineSize();

    public abstract boolean engineIsKeyEntry(String alias);

    public abstract boolean engineIsCertificateEntry(String alias);

    public abstract String engineGetCertificateAlias(Certificate cert);

    public abstract void engineStore(OutputStream stream, char[] password) throws IOException, NoSuchAlgorithmException, CertificateException;

    public void engineStore(KeyStore.LoadStoreParameter param) throws IOException, NoSuchAlgorithmException, CertificateException;

    public abstract void engineLoad(InputStream stream, char[] password) throws IOException, NoSuchAlgorithmException, CertificateException;

    public void engineLoad(KeyStore.LoadStoreParameter param) throws IOException, NoSuchAlgorithmException, CertificateException;

    void engineLoad(InputStream stream, KeyStore.LoadStoreParameter param) throws IOException, NoSuchAlgorithmException, CertificateException;

    public KeyStore.Entry engineGetEntry(String alias, KeyStore.ProtectionParameter protParam) throws KeyStoreException, NoSuchAlgorithmException, UnrecoverableEntryException;

    public void engineSetEntry(String alias, KeyStore.Entry entry, KeyStore.ProtectionParameter protParam) throws KeyStoreException;

    public boolean engineEntryInstanceOf(String alias, Class<? extends KeyStore.Entry> entryClass);

    public boolean engineProbe(InputStream stream) throws IOException;
}
