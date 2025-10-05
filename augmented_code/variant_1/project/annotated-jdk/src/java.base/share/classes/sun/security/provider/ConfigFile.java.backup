/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.provider;

import java.io.*;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.security.Security;
import java.security.URIParameter;
import java.text.MessageFormat;
import java.util.*;
import javax.security.auth.AuthPermission;
import javax.security.auth.login.AppConfigurationEntry;
import javax.security.auth.login.AppConfigurationEntry.LoginModuleControlFlag;
import javax.security.auth.login.Configuration;
import javax.security.auth.login.ConfigurationSpi;
import sun.security.util.Debug;
import sun.security.util.PropertyExpander;
import sun.security.util.ResourcesMgr;
import static java.nio.charset.StandardCharsets.UTF_8;
import org.checkerframework.dataflow.qual.Pure;

public final class ConfigFile extends Configuration {

    public ConfigFile() {
    }

    @Override
    public AppConfigurationEntry[] getAppConfigurationEntry(String appName);

    @Override
    public synchronized void refresh();

    public static final class Spi extends ConfigurationSpi {

        public Spi() {
        }

        public Spi(URI uri) {
        }

        @SuppressWarnings("removal")
        public Spi(final Configuration.Parameters params) throws IOException {
        }

        @Override
        public AppConfigurationEntry[] engineGetAppConfigurationEntry(String applicationName);

        @SuppressWarnings("removal")
        @Override
        public synchronized void engineRefresh();
    }
}
