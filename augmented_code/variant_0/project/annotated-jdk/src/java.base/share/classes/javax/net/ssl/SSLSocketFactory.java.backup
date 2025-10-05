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
package javax.net.ssl;

import org.checkerframework.checker.mustcall.qual.MustCallAlias;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.net.*;
import javax.net.SocketFactory;
import java.io.IOException;
import java.io.InputStream;
import java.security.*;
import java.util.Locale;
import sun.security.action.GetPropertyAction;

@AnnotatedFor({ "mustcall" })
public abstract class SSLSocketFactory extends SocketFactory {

    public SSLSocketFactory() {
    }

    public static SocketFactory getDefault();

    @SuppressWarnings("removal")
    static String getSecurityProperty(final String name);

    public abstract String[] getDefaultCipherSuites();

    public abstract String[] getSupportedCipherSuites();

    @MustCallAlias
    public abstract Socket createSocket(@MustCallAlias Socket s, String host, int port, boolean autoClose) throws IOException;

    @MustCallAlias
    public Socket createSocket(@MustCallAlias Socket s, InputStream consumed, boolean autoClose) throws IOException;

    private static final class DefaultFactoryHolder {
    }
}

class DefaultSSLSocketFactory extends SSLSocketFactory {

    @Override
    public Socket createSocket() throws IOException;

    @Override
    public Socket createSocket(String host, int port) throws IOException;

    @Override
    public Socket createSocket(Socket s, String host, int port, boolean autoClose) throws IOException;

    @Override
    public Socket createSocket(InetAddress address, int port) throws IOException;

    @Override
    public Socket createSocket(String host, int port, InetAddress clientAddress, int clientPort) throws IOException;

    @Override
    public Socket createSocket(InetAddress address, int port, InetAddress clientAddress, int clientPort) throws IOException;

    @Override
    public String[] getDefaultCipherSuites();

    @Override
    public String[] getSupportedCipherSuites();
}
