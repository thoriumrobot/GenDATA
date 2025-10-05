/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.tools.keytool;

import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.*;
import java.security.cert.Certificate;
import java.security.cert.CertificateFactory;
import java.security.cert.CertStoreException;
import java.security.cert.CRL;
import java.security.cert.X509Certificate;
import java.security.cert.CertificateException;
import java.security.cert.URICertStoreParameters;
import java.security.interfaces.ECKey;
import java.security.interfaces.EdECKey;
import java.security.spec.ECParameterSpec;
import java.text.Collator;
import java.text.MessageFormat;
import java.util.*;
import java.util.function.BiFunction;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.math.BigInteger;
import java.net.URI;
import java.net.URL;
import java.net.URLClassLoader;
import java.security.cert.CertStore;
import java.security.cert.X509CRL;
import java.security.cert.X509CRLEntry;
import java.security.cert.X509CRLSelector;
import javax.security.auth.x500.X500Principal;
import java.util.Base64;
import sun.security.pkcs12.PKCS12KeyStore;
import sun.security.util.ECKeySizeParameterSpec;
import sun.security.util.KeyUtil;
import sun.security.util.NamedCurve;
import sun.security.util.ObjectIdentifier;
import sun.security.pkcs10.PKCS10;
import sun.security.pkcs10.PKCS10Attribute;
import sun.security.provider.X509Factory;
import sun.security.provider.certpath.ssl.SSLServerCertStore;
import sun.security.util.KnownOIDs;
import sun.security.util.Password;
import sun.security.util.SecurityProperties;
import sun.security.util.SecurityProviderConstants;
import sun.security.util.SignatureUtil;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.PBEKeySpec;
import sun.security.pkcs.PKCS9Attribute;
import sun.security.tools.KeyStoreUtil;
import sun.security.tools.PathList;
import sun.security.util.DerValue;
import sun.security.util.Pem;
import sun.security.x509.*;
import static java.security.KeyStore.*;
import static sun.security.tools.keytool.Main.Command.*;
import static sun.security.tools.keytool.Main.Option.*;
import sun.security.util.DisabledAlgorithmConstraints;

public final class Main {

    public static void main(String[] args) throws Exception;

    String[] parseArgs(String[] args) throws Exception;

    boolean isKeyStoreRelated(Command cmd);

    void doCommands(PrintStream out) throws Exception;

    boolean inplaceImportCheck() throws Exception;

    KeyStore loadSourceKeyStore() throws Exception;

    public static Collection<? extends CRL> loadCRLs(String src) throws Exception;

    public static List<CRL> readCRLsFromCert(X509Certificate cert) throws Exception;
}

class Pair<A, B> {

    public final A fst;

    public final B snd;

    public Pair(A fst, B snd) {
    }

    public String toString();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);

    public int hashCode();

    public static <A, B> Pair<A, B> of(A a, B b);
}
