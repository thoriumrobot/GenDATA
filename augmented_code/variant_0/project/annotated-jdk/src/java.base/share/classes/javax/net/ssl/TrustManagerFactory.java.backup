/*
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.net.ssl;

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.security.Security;
import java.security.*;
import java.util.Objects;
import sun.security.jca.GetInstance;

@AnnotatedFor("nullness")
public class TrustManagerFactory {

    @SuppressWarnings("removal")
    public static final String getDefaultAlgorithm();

    protected TrustManagerFactory(TrustManagerFactorySpi factorySpi, Provider provider, String algorithm) {
    }

    public final String getAlgorithm();

    public static final TrustManagerFactory getInstance(String algorithm) throws NoSuchAlgorithmException;

    public static final TrustManagerFactory getInstance(String algorithm, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    public static final TrustManagerFactory getInstance(String algorithm, Provider provider) throws NoSuchAlgorithmException;

    public final Provider getProvider();

    public final void init(@Nullable KeyStore ks) throws KeyStoreException;

    public final void init(ManagerFactoryParameters spec) throws InvalidAlgorithmParameterException;

    public final TrustManager[] getTrustManagers();
}
