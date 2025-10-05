/*
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.pkcs11;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.*;
import java.util.*;
import java.security.*;
import java.security.interfaces.*;
import javax.crypto.interfaces.*;
import javax.security.auth.Subject;
import javax.security.auth.login.LoginException;
import javax.security.auth.login.FailedLoginException;
import javax.security.auth.callback.Callback;
import javax.security.auth.callback.CallbackHandler;
import javax.security.auth.callback.PasswordCallback;
import com.sun.crypto.provider.ChaCha20Poly1305Parameters;
import sun.security.util.Debug;
import sun.security.util.ResourcesMgr;
import static sun.security.util.SecurityConstants.PROVIDER_VER;
import static sun.security.util.SecurityProviderConstants.getAliases;
import sun.security.pkcs11.Secmod.*;
import sun.security.pkcs11.wrapper.*;
import static sun.security.pkcs11.wrapper.PKCS11Constants.*;
import static sun.security.pkcs11.wrapper.PKCS11Exception.*;

public final class SunPKCS11 extends AuthProvider {

    Token getToken();

    public SunPKCS11() {
    }

    @SuppressWarnings("removal")
    @Override
    public Provider configure(String configArg) throws InvalidParameterException;

    @Override
    public boolean isConfigured();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    private static final class Descriptor {

        public String toString();
    }

    private static class TokenPoller extends Thread {

        @Override
        public void run();

        void disable();
    }

    private class NativeResourceCleaner extends Thread {

        @Override
        public void run();
    }

    @SuppressWarnings("removal")
    synchronized void uninitToken(Token token);

    private static final class P11Service extends Service {

        @Override
        public Object newInstance(Object param) throws NoSuchAlgorithmException;

        public Object newInstance0(Object param) throws PKCS11Exception, NoSuchAlgorithmException;

        public boolean supportsParameter(Object param);

        public String toString();
    }

    public void login(Subject subject, CallbackHandler handler) throws LoginException;

    public void logout() throws LoginException;

    public void setCallbackHandler(CallbackHandler handler);

    private static class SunPKCS11Rep implements Serializable {
    }
}
