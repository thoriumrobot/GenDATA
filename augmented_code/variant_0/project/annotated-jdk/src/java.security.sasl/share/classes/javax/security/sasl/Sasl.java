/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.security.sasl;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import javax.security.auth.callback.CallbackHandler;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Collections;
    @Positive
import java.security.InvalidParameterException;
    @Positive
import java.security.NoSuchAlgorithmException;
    @Positive
import java.security.Provider;
    @Positive
import java.security.Provider.Service;
    @Positive
import java.security.Security;
    @Positive
import java.util.logging.Level;
    @Positive
import java.util.logging.Logger;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public class Sasl {

    @Positive
    @Interned
    @Positive
    public static final String QOP;

    @Positive
    @Interned
    @Positive
    public static final String STRENGTH;

    @Positive
    public static final String SERVER_AUTH;

    @Positive
    public static final String BOUND_SERVER_NAME;

    @Positive
    @Interned
    @Positive
    public static final String MAX_BUFFER;

    @Positive
    @Interned
    @Positive
    public static final String RAW_SEND_SIZE;

    @Positive
    @Interned
    @Positive
    public static final String REUSE;

    @Positive
    public static final String POLICY_NOPLAINTEXT;

    @Positive
    public static final String POLICY_NOACTIVE;

    @Positive
    public static final String POLICY_NODICTIONARY;

    @Positive
    public static final String POLICY_NOANONYMOUS;

    @Positive
    public static final String POLICY_FORWARD_SECRECY;

    @Positive
    public static final String POLICY_PASS_CREDENTIALS;

    @Positive
    @Interned
    @Positive
    public static final String CREDENTIALS;

    @Positive
    public static SaslClient createSaslClient(String[] mechanisms, String authorizationId, String protocol, String serverName, Map<String, ?> props, CallbackHandler cbh) throws SaslException;

    @Positive
    public static SaslServer createSaslServer(String mechanism, String protocol, String serverName, Map<String, ?> props, javax.security.auth.callback.CallbackHandler cbh) throws SaslException;

    @Positive
    public static Enumeration<SaslClientFactory> getSaslClientFactories();

    @Positive
    public static Enumeration<SaslServerFactory> getSaslServerFactories();
    @Positive
}
