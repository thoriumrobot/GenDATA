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
package javax.security.sasl;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import javax.security.auth.callback.CallbackHandler;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.Collections;
import java.security.InvalidParameterException;
import java.security.NoSuchAlgorithmException;
import java.security.Provider;
import java.security.Provider.Service;
import java.security.Security;
import java.util.logging.Level;
import java.util.logging.Logger;

@AnnotatedFor({ "interning" })
public class Sasl {

    @Interned
    public static final String QOP;

    @Interned
    public static final String STRENGTH;

    public static final String SERVER_AUTH;

    public static final String BOUND_SERVER_NAME;

    @Interned
    public static final String MAX_BUFFER;

    @Interned
    public static final String RAW_SEND_SIZE;

    @Interned
    public static final String REUSE;

    public static final String POLICY_NOPLAINTEXT;

    public static final String POLICY_NOACTIVE;

    public static final String POLICY_NODICTIONARY;

    public static final String POLICY_NOANONYMOUS;

    public static final String POLICY_FORWARD_SECRECY;

    public static final String POLICY_PASS_CREDENTIALS;

    @Interned
    public static final String CREDENTIALS;

    public static SaslClient createSaslClient(String[] mechanisms, String authorizationId, String protocol, String serverName, Map<String, ?> props, CallbackHandler cbh) throws SaslException;

    public static SaslServer createSaslServer(String mechanism, String protocol, String serverName, Map<String, ?> props, javax.security.auth.callback.CallbackHandler cbh) throws SaslException;

    public static Enumeration<SaslClientFactory> getSaslClientFactories();

    public static Enumeration<SaslServerFactory> getSaslServerFactories();
}
