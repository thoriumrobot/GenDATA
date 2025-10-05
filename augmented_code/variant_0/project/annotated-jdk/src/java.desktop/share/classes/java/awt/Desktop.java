/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

    @Positive
import org.checkerframework.checker.guieffect.qual.SafeEffect;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.desktop.AboutEvent;
    @Positive
import java.awt.desktop.AboutHandler;
    @Positive
import java.awt.desktop.OpenFilesHandler;
    @Positive
import java.awt.desktop.OpenURIEvent;
    @Positive
import java.awt.desktop.OpenURIHandler;
    @Positive
import java.awt.desktop.PreferencesEvent;
    @Positive
import java.awt.desktop.PreferencesHandler;
    @Positive
import java.awt.desktop.PrintFilesHandler;
    @Positive
import java.awt.desktop.QuitHandler;
    @Positive
import java.awt.desktop.QuitStrategy;
    @Positive
import java.awt.desktop.SystemEventListener;
    @Positive
import java.awt.peer.DesktopPeer;
    @Positive
import java.io.File;
    @Positive
import java.io.FilePermission;
    @Positive
import java.io.IOException;
    @Positive
import java.net.URI;
    @Positive
import java.net.URISyntaxException;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Objects;
    @Positive
import javax.swing.JMenuBar;
    @Positive
import sun.awt.SunToolkit;
    @Positive
import sun.security.util.SecurityConstants;

    @Positive
@AnnotatedFor({ "guieffect", "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class Desktop {

    @Positive
    public static enum Action {

    @Positive
        OPEN,
    @Positive
        EDIT,
    @Positive
        PRINT,
    @Positive
        MAIL,
    @Positive
        BROWSE,
    @Positive
        APP_EVENT_FOREGROUND,
    @Positive
        APP_EVENT_HIDDEN,
    @Positive
        APP_EVENT_REOPENED,
    @Positive
        APP_EVENT_SCREEN_SLEEP,
    @Positive
        APP_EVENT_SYSTEM_SLEEP,
    @Positive
        APP_EVENT_USER_SESSION,
    @Positive
        APP_ABOUT,
    @Positive
        APP_PREFERENCES,
    @Positive
        APP_OPEN_FILE,
    @Positive
        APP_PRINT_FILE,
    @Positive
        APP_OPEN_URI,
    @Positive
        APP_QUIT_HANDLER,
    @Positive
        APP_QUIT_STRATEGY,
    @Positive
        APP_SUDDEN_TERMINATION,
    @Positive
        APP_REQUEST_FOREGROUND,
    @Positive
        APP_HELP_VIEWER,
    @Positive
        APP_MENU_BAR,
    @Positive
        BROWSE_FILE_DIR,
    @Positive
        MOVE_TO_TRASH
    @Positive
    }

    @Positive
    @SafeEffect
    @Positive
    public static synchronized Desktop getDesktop();

    @Positive
    @SafeEffect
    @Positive
    public static boolean isDesktopSupported();

    @Positive
    @SafeEffect
    @Positive
    public boolean isSupported(Action action);

    @Positive
    @SafeEffect
    @Positive
    public void open(File file) throws IOException;

    @Positive
    @SafeEffect
    @Positive
    public void edit(File file) throws IOException;

    @Positive
    @SafeEffect
    @Positive
    public void print(File file) throws IOException;

    @Positive
    @SafeEffect
    @Positive
    public void browse(URI uri) throws IOException;

    @Positive
    @SafeEffect
    @Positive
    public void mail() throws IOException;

    @Positive
    @SafeEffect
    @Positive
    public void mail(URI mailtoURI) throws IOException;

    @Positive
    public void addAppEventListener(final SystemEventListener listener);

    @Positive
    public void removeAppEventListener(final SystemEventListener listener);

    @Positive
    public void setAboutHandler(final AboutHandler aboutHandler);

    @Positive
    public void setPreferencesHandler(final PreferencesHandler preferencesHandler);

    @Positive
    public void setOpenFileHandler(final OpenFilesHandler openFileHandler);

    @Positive
    public void setPrintFileHandler(final PrintFilesHandler printFileHandler);

    @Positive
    public void setOpenURIHandler(final OpenURIHandler openURIHandler);

    @Positive
    public void setQuitHandler(final QuitHandler quitHandler);

    @Positive
    public void setQuitStrategy(final QuitStrategy strategy);

    @Positive
    public void enableSuddenTermination();

    @Positive
    public void disableSuddenTermination();

    @Positive
    public void requestForeground(final boolean allWindows);

    @Positive
    public void openHelpViewer();

    @Positive
    public void setDefaultMenuBar(final JMenuBar menuBar);

    @Positive
    public void browseFileDirectory(File file);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public boolean moveToTrash(File file);
    @Positive
}
