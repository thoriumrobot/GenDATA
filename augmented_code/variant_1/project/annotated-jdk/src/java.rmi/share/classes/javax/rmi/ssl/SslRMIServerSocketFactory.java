/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2003, 2008, Oracle and/or its affiliates. All rights reserved.
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
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
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
package javax.rmi.ssl;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.IOException;
    @Positive
import java.net.ServerSocket;
    @Positive
import java.net.Socket;
    @Positive
import java.rmi.server.RMIServerSocketFactory;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.List;
    @Positive
import javax.net.ssl.SSLContext;
    @Positive
import javax.net.ssl.SSLServerSocketFactory;
    @Positive
import javax.net.ssl.SSLSocket;
    @Positive
import javax.net.ssl.SSLSocketFactory;

    @Positive
public class SslRMIServerSocketFactory implements RMIServerSocketFactory {

    @Positive
    public SslRMIServerSocketFactory() {
    @Positive
    }

    @Positive
    public SslRMIServerSocketFactory(String[] enabledCipherSuites, String[] enabledProtocols, boolean needClientAuth) throws IllegalArgumentException {
    @Positive
    }

    @Positive
    public SslRMIServerSocketFactory(SSLContext context, String[] enabledCipherSuites, String[] enabledProtocols, boolean needClientAuth) throws IllegalArgumentException {
    @Positive
    }

    @Positive
    public final String[] getEnabledCipherSuites();

    @Positive
    public final String[] getEnabledProtocols();

    @Positive
    public final boolean getNeedClientAuth();

    @Positive
    public ServerSocket createServerSocket(int port) throws IOException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();
    @Positive
}
