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
package java.net;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import sun.net.www.protocol.http.AuthenticatorKeys;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class Authenticator {

    @Positive
    public Authenticator() {
    @Positive
    }

    @Positive
    public enum RequestorType {

    @Positive
        PROXY, SERVER
    @Positive
    }

    @Positive
    public static synchronized void setDefault(Authenticator a);

    @Positive
    public static Authenticator getDefault();

    @Positive
    public static PasswordAuthentication requestPasswordAuthentication(InetAddress addr, int port, String protocol, String prompt, String scheme);

    @Positive
    public static PasswordAuthentication requestPasswordAuthentication(String host, InetAddress addr, int port, String protocol, String prompt, String scheme);

    @Positive
    public static PasswordAuthentication requestPasswordAuthentication(String host, InetAddress addr, int port, String protocol, String prompt, String scheme, URL url, RequestorType reqType);

    @Positive
    public static PasswordAuthentication requestPasswordAuthentication(Authenticator authenticator, String host, InetAddress addr, int port, String protocol, String prompt, String scheme, URL url, RequestorType reqType);

    @Positive
    public PasswordAuthentication requestPasswordAuthenticationInstance(String host, InetAddress addr, int port, String protocol, String prompt, String scheme, URL url, RequestorType reqType);

    @Positive
    protected final String getRequestingHost();

    @Positive
    protected final InetAddress getRequestingSite();

    @Positive
    protected final int getRequestingPort();

    @Positive
    protected final String getRequestingProtocol();

    @Positive
    protected final String getRequestingPrompt();

    @Positive
    protected final String getRequestingScheme();

    @Positive
    protected PasswordAuthentication getPasswordAuthentication();

    @Positive
    protected URL getRequestingURL();

    @Positive
    protected RequestorType getRequestorType();

    @Positive
    static String getKey(Authenticator a);
    @Positive
}

// CFWR semantic augmentation - variant 0
