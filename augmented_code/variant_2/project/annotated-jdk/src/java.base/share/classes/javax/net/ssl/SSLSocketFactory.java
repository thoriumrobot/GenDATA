/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package javax.net.ssl;

    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.net.*;
    @Positive
import javax.net.SocketFactory;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.security.*;
    @Positive
import java.util.Locale;
    @Positive
import sun.security.action.GetPropertyAction;

    @Positive
@AnnotatedFor({ "mustcall" })
    @Positive
public abstract class SSLSocketFactory extends SocketFactory {

    @Positive
    public SSLSocketFactory() {
    @Positive
    }

    @Positive
    public static SocketFactory getDefault();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    static String getSecurityProperty(final String name);

    @Positive
    public abstract String[] getDefaultCipherSuites();

    @Positive
    public abstract String[] getSupportedCipherSuites();

    @Positive
    @MustCallAlias
    @Positive
    public abstract Socket createSocket(@MustCallAlias Socket s, String host, int port, boolean autoClose) throws IOException;

    @Positive
    @MustCallAlias
    @Positive
    public Socket createSocket(@MustCallAlias Socket s, InputStream consumed, boolean autoClose) throws IOException;

    @Positive
    private static final class DefaultFactoryHolder {
    @Positive
    }
    @Positive
}

    @Positive
class DefaultSSLSocketFactory extends SSLSocketFactory {

    @Positive
    @Override
    @Positive
    public Socket createSocket() throws IOException;

    @Positive
    @Override
    @Positive
    public Socket createSocket(String host, int port) throws IOException;

    @Positive
    @Override
    @Positive
    public Socket createSocket(Socket s, String host, int port, boolean autoClose) throws IOException;

    @Positive
    @Override
    @Positive
    public Socket createSocket(InetAddress address, int port) throws IOException;

    @Positive
    @Override
    @Positive
    public Socket createSocket(String host, int port, InetAddress clientAddress, int clientPort) throws IOException;

    @Positive
    @Override
    @Positive
    public Socket createSocket(InetAddress address, int port, InetAddress clientAddress, int clientPort) throws IOException;

    @Positive
    @Override
    @Positive
    public String[] getDefaultCipherSuites();

    @Positive
    @Override
    @Positive
    public String[] getSupportedCipherSuites();
    @Positive
}
