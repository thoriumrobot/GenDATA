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
package java.net;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import sun.net.www.protocol.http.AuthenticatorKeys;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class Authenticator {

    public Authenticator() {
    }

    public enum RequestorType {

        PROXY, SERVER
    }

    public static synchronized void setDefault(Authenticator a);

    public static Authenticator getDefault();

    public static PasswordAuthentication requestPasswordAuthentication(InetAddress addr, int port, String protocol, String prompt, String scheme);

    public static PasswordAuthentication requestPasswordAuthentication(String host, InetAddress addr, int port, String protocol, String prompt, String scheme);

    public static PasswordAuthentication requestPasswordAuthentication(String host, InetAddress addr, int port, String protocol, String prompt, String scheme, URL url, RequestorType reqType);

    public static PasswordAuthentication requestPasswordAuthentication(Authenticator authenticator, String host, InetAddress addr, int port, String protocol, String prompt, String scheme, URL url, RequestorType reqType);

    public PasswordAuthentication requestPasswordAuthenticationInstance(String host, InetAddress addr, int port, String protocol, String prompt, String scheme, URL url, RequestorType reqType);

    protected final String getRequestingHost();

    protected final InetAddress getRequestingSite();

    protected final int getRequestingPort();

    protected final String getRequestingProtocol();

    protected final String getRequestingPrompt();

    protected final String getRequestingScheme();

    protected PasswordAuthentication getPasswordAuthentication();

    protected URL getRequestingURL();

    protected RequestorType getRequestorType();

    static String getKey(Authenticator a);
}
