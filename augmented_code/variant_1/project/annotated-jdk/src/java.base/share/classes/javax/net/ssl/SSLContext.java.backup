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

import org.checkerframework.checker.nullness.qual.MonotonicNonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.security.*;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Objects;
import sun.security.jca.GetInstance;

@AnnotatedFor("nullness")
public class SSLContext {

    protected SSLContext(SSLContextSpi contextSpi, Provider provider, String protocol) {
    }

    public static SSLContext getDefault() throws NoSuchAlgorithmException;

    public static void setDefault(SSLContext context);

    public static SSLContext getInstance(String protocol) throws NoSuchAlgorithmException;

    public static SSLContext getInstance(String protocol, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    public static SSLContext getInstance(String protocol, Provider provider) throws NoSuchAlgorithmException;

    public final String getProtocol();

    public final Provider getProvider();

    public final void init(KeyManager @Nullable [] km, TrustManager @Nullable [] tm, @Nullable SecureRandom random) throws KeyManagementException;

    public final SSLSocketFactory getSocketFactory();

    public final SSLServerSocketFactory getServerSocketFactory();

    public final SSLEngine createSSLEngine();

    public final SSLEngine createSSLEngine(String peerHost, int peerPort);

    @Nullable
    public final SSLSessionContext getServerSessionContext();

    @Nullable
    public final SSLSessionContext getClientSessionContext();

    public final SSLParameters getDefaultSSLParameters();

    public final SSLParameters getSupportedSSLParameters();
}
