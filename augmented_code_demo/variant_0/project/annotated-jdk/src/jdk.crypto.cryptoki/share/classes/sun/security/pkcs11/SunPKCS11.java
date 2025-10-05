/*
    @Positive
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.pkcs11;

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
import java.io.*;
    @Positive
import java.util.*;
    @Positive
import java.security.*;
    @Positive
import java.security.interfaces.*;
    @Positive
import javax.crypto.interfaces.*;
    @Positive
import javax.security.auth.Subject;
    @Positive
import javax.security.auth.login.LoginException;
    @Positive
import javax.security.auth.login.FailedLoginException;
    @Positive
import javax.security.auth.callback.Callback;
    @Positive
import javax.security.auth.callback.CallbackHandler;
    @Positive
import javax.security.auth.callback.PasswordCallback;
    @Positive
import com.sun.crypto.provider.ChaCha20Poly1305Parameters;
    @Positive
import sun.security.util.Debug;
    @Positive
import sun.security.util.ResourcesMgr;
    @Positive
import static sun.security.util.SecurityConstants.PROVIDER_VER;
    @Positive
import static sun.security.util.SecurityProviderConstants.getAliases;
    @Positive
import sun.security.pkcs11.Secmod.*;
    @Positive
import sun.security.pkcs11.wrapper.*;
    @Positive
import static sun.security.pkcs11.wrapper.PKCS11Constants.*;
    @Positive
import static sun.security.pkcs11.wrapper.PKCS11Exception.*;

    @Positive
public final class SunPKCS11 extends AuthProvider {

    @Positive
    Token getToken();

    @Positive
    public SunPKCS11() {
    @Positive
    }

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @Override
    @Positive
    public Provider configure(String configArg) throws InvalidParameterException;

    @Positive
    @Override
    @Positive
    public boolean isConfigured();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    private static final class Descriptor {

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    private static class TokenPoller extends Thread {

    @Positive
        @Override
    @Positive
        public void run();

    @Positive
        void disable();
    @Positive
    }

    @Positive
    private class NativeResourceCleaner extends Thread {

    @Positive
        @Override
    @Positive
        public void run();
    @Positive
    }

    @Positive
    @SuppressWarnings("removal")
    @Positive
    synchronized void uninitToken(Token token);

    @Positive
    private static final class P11Service extends Service {

    @Positive
        @Override
    @Positive
        public Object newInstance(Object param) throws NoSuchAlgorithmException;

    @Positive
        public Object newInstance0(Object param) throws PKCS11Exception, NoSuchAlgorithmException;

    @Positive
        public boolean supportsParameter(Object param);

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public void login(Subject subject, CallbackHandler handler) throws LoginException;

    @Positive
    public void logout() throws LoginException;

    @Positive
    public void setCallbackHandler(CallbackHandler handler);

    @Positive
    private static class SunPKCS11Rep implements Serializable {
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
