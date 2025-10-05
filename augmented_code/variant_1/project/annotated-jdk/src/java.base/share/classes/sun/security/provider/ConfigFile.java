/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package sun.security.provider;

    @Positive
import java.io.*;
    @Positive
import java.net.MalformedURLException;
    @Positive
import java.net.URI;
    @Positive
import java.net.URL;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.PrivilegedActionException;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.security.Security;
    @Positive
import java.security.URIParameter;
    @Positive
import java.text.MessageFormat;
    @Positive
import java.util.*;
    @Positive
import javax.security.auth.AuthPermission;
    @Positive
import javax.security.auth.login.AppConfigurationEntry;
    @Positive
import javax.security.auth.login.AppConfigurationEntry.LoginModuleControlFlag;
    @Positive
import javax.security.auth.login.Configuration;
    @Positive
import javax.security.auth.login.ConfigurationSpi;
    @Positive
import sun.security.util.Debug;
    @Positive
import sun.security.util.PropertyExpander;
    @Positive
import sun.security.util.ResourcesMgr;
    @Positive
import static java.nio.charset.StandardCharsets.UTF_8;
    @Positive
import org.checkerframework.dataflow.qual.Pure;

    @Positive
public final class ConfigFile extends Configuration {

    @Positive
    public ConfigFile() {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public AppConfigurationEntry[] getAppConfigurationEntry(String appName);

    @Positive
    @Override
    @Positive
    public synchronized void refresh();

    @Positive
    public static final class Spi extends ConfigurationSpi {

    @Positive
        public Spi() {
    @Positive
        }

    @Positive
        public Spi(URI uri) {
    @Positive
        }

    @Positive
        @SuppressWarnings("removal")
    @Positive
        public Spi(final Configuration.Parameters params) throws IOException {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public AppConfigurationEntry[] engineGetAppConfigurationEntry(String applicationName);

    @Positive
        @SuppressWarnings("removal")
    @Positive
        @Override
    @Positive
        public synchronized void engineRefresh();
    @Positive
    }
    @Positive
}
