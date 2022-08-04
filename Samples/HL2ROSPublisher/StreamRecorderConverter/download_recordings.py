import os
import cmd
from urllib import request
from urllib.parse import quote
import json
import argparse
from glob import glob

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "HL2ROSPublisher"


class MyShell(cmd.Cmd):
    dev_portal_browser = None
    save_path = None

    # welcome info
    intro = 'Welcome to MyShell, Type "help" or "?" to list commands.'
    prompt = "(myshell)>>> "
    ruler = "-"

    def __init__(self, save_path, dev_portal_browser):
        super().__init__()
        self.dev_portal_browser = dev_portal_browser
        self.save_path = save_path

    def do_help(self, arg):
        print_help()

    def do_exit(self, arg):
        return True

    def do_list(self, arg):
        print("  * Recordings on Hololens:")
        self.dev_portal_browser.list_recordings()
        print("  * Recordings on save folder:")
        list_workspace_recordings(self.save_path)

    def do_list_device(self, arg):
        self.dev_portal_browser.list_recordings()

    def do_list_workspace(self, arg):
        list_workspace_recordings(self.save_path)

    def do_download(self, arg):
        try:
            recording_idx = int(arg)
            if recording_idx is not None:
                self.dev_portal_browser.download_recording(
                    recording_idx, self.save_path
                )
        except ValueError:
            print(f"Cannot download {arg}")

    def do_download_all(self, arg):
        for recording_idx in range(len(self.dev_portal_browser.recording_names)):
            self.dev_portal_browser.download_recording(recording_idx, self.save_path)

    def do_delete(self, arg):
        try:
            recording_idx = int(arg)
            if recording_idx is not None:
                self.dev_portal_browser.delete_recording(recording_idx)
        except ValueError:
            print(f"Cannot delete {arg}")

    def do_delete_all(self, arg):
        for _ in range(len(self.dev_portal_browser.recording_names)):
            self.dev_portal_browser.delete_recording(0)


class DevicePortalBrowser(object):
    def connect(self, address, username, password):
        print("==> Connecting to Hololens Device Portal...")
        self.url = f"http://{address}"
        password_manager = request.HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(None, self.url, username, password)
        handler = request.HTTPBasicAuthHandler(password_manager)
        opener = request.build_opener(handler)
        opener.open(self.url)
        request.install_opener(opener)
        print(f"  * Connected to Hololens at Address: {self.url}")

        print(f"==> Searching for {APP_NAME}...")
        response = request.urlopen(f"{self.url}/api/app/packagemanager/packages")
        packages = json.loads(response.read().decode())
        self.package_full_name = None
        for package in packages["InstalledPackages"]:
            if package["Name"] == APP_NAME:
                self.package_full_name = package["PackageFullName"]
                break
        assert (
            self.package_full_name is not None
        ), f"ERROR: Package {APP_NAME} is not installed!!!"
        print(f'  * Found {APP_NAME} application with name: "{self.package_full_name}"')

        print("==> Searching for recordings...")
        request_url = (
            f"{self.url}/api/filesystem/apps/files?knownfolderid=LocalAppData"
            + f"&packagefullname={quote(self.package_full_name)}"
            + f"&path=/LocalState"
        )
        response = request.urlopen(request_url)
        recordings = json.loads(response.read().decode())
        self.recording_names = []
        for recording in recordings["Items"]:
            # Check if the recording contains any file data.
            recording_id = recording["Id"]
            request_url = (
                f"{self.url}/api/filesystem/apps/files?knownfolderid=LocalAppData"
                + f"&packagefullname={self.package_full_name}"
                + f"&path=/LocalState/{recording_id}"
            )
            response = request.urlopen(request_url)
            files = json.loads(response.read().decode())
            if len(files["Items"]) > 0:
                self.recording_names.append(recording["Id"])
        self.recording_names.sort()
        print(f"  * Found a total of {len(self.recording_names)} recordings")

    def list_recordings(self):
        response = request.urlopen(
            f"{self.url}/api/filesystem/apps/files?knownfolderid=LocalAppData"
            + f"&packagefullname={quote(self.package_full_name)}"
            + f"&path=/LocalState"
        )
        recordings = json.loads(response.read().decode())
        self.recording_names = []
        for recording in recordings["Items"]:
            # Check if the recording contains any file data.
            recording_id = recording["Id"]
            request_url = (
                f"{self.url}/api/filesystem/apps/files?knownfolderid=LocalAppData"
                + f"&packagefullname={self.package_full_name}"
                + f"&path=/LocalState/{recording_id}"
            )
            response = request.urlopen(request_url)
            files = json.loads(response.read().decode())
            if len(files["Items"]) > 0:
                self.recording_names.append(recording["Id"])
        self.recording_names.sort()

        for i, recording_name in enumerate(self.recording_names):
            print(f"    [{i:6d}]  {recording_name}")

        if len(self.recording_names) == 0:
            print("  * No recordings found on device")

    def get_recording_name(self, recording_idx):
        try:
            return self.recording_names[recording_idx]
        except IndexError:
            print("  * Recording does not exist")

    def download_recording(self, recording_idx, w_path):
        recording_name = self.get_recording_name(recording_idx)
        if recording_name is None:
            return

        recording_path = os.path.join(w_path, recording_name)
        os.makedirs(recording_path, exist_ok=True)

        print("Downloading recording {}...".format(recording_name))

        response = request.urlopen(
            f"{self.url}/api/filesystem/apps/files?knownfolderid=LocalAppData"
            + f"&packagefullname={self.package_full_name}"
            + f"&path=/LocalState/{recording_name}"
        )
        files = json.loads(response.read().decode())

        for file in files["Items"]:
            if file["Type"] != 32:
                continue

            destination_path = os.path.join(recording_path, file["Id"])
            if os.path.exists(destination_path):
                print("  * Skipping, already downloaded:", file["Id"])
                continue

            print("  * Downloading:", file["Id"])
            request.urlretrieve(
                f"{self.url}/api/filesystem/apps/file?knownfolderid=LocalAppData"
                + f"&packagefullname={self.package_full_name}"
                + f"&filename=/LocalState/{recording_name}/{quote(file['Id'])}",
                str(destination_path),
            )

    def delete_recording(self, recording_idx):
        recording_name = self.get_recording_name(recording_idx)
        if recording_name is None:
            return

        print("Deleting recording {}...".format(recording_name))

        response = request.urlopen(
            f"{self.url}/api/filesystem/apps/files?knownfolderid="
            + f"LocalAppData&packagefullname={self.package_full_name}"
            + f"&path=/LocalState/{recording_name}"
        )
        files = json.loads(response.read().decode())

        for file in files["Items"]:
            if file["Type"] != 32:
                continue

            print("=> Deleting:", file["Id"])
            request.urlopen(
                request.Request(
                    f"{self.url}/api/filesystem/apps/file?knownfolderid=LocalAppData"
                    + f"&packagefullname={self.package_full_name}"
                    + f"&filename=/LocalState/{recording_name}/{quote(file['Id'])}",
                    method="DELETE",
                )
            )

        self.recording_names.remove(recording_name)


def print_help():
    print("Available commands:")
    print("  help:                     Print this help message")
    print("  exit:                     Exit the console loop")
    print("  list:                     List all recordings")
    print("  list_device:              List all recordings on the HoloLens")
    print("  list_workspace:           List all recordings in the workspace")
    print("  download X:               Download recording X from the HoloLens")
    print("  download_all:             Download all recordings from the HoloLens")
    print("  delete X:                 Delete recording X from the HoloLens")
    print("  delete_all:               Delete all recordings from the HoloLens")


def list_workspace_recordings(w_path):
    recording_names = sorted(glob(os.path.join(w_path, "*")))
    for i, recording_name in enumerate(recording_names):
        name = os.path.split(recording_name)[-1]
        print("    [{: 6d}]  {}".format(i, name))
    if len(recording_names) == 0:
        print("    * No recordings found in workspace")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        # default="127.0.0.1:10080", this is used when connect with usb-c cable
        default="192.168.50.210:80",
        help="The IP address for the HoloLens Device Portal",
    )
    parser.add_argument(
        "--username",
        # required=True,
        default="admin",
        help="The username for the HoloLens Device Portal",
    )
    parser.add_argument(
        "--password",
        # required=True,
        default="123456789",
        help="The password for the HoloLens Device Portal",
    )
    parser.add_argument(
        "--save_path",
        # required=True,
        default=os.path.join(CURR_DIR, "Recordings"),
        help="Path to workspace folder used for downloading recordings",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    portal_address = args.address
    portal_username = args.username
    portal_password = args.password
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    dev_portal_browser = DevicePortalBrowser()
    dev_portal_browser.connect(
        address=portal_address, username=portal_username, password=portal_password
    )

    print()
    print_help()
    print()

    dev_portal_browser.list_recordings()
    mysh = MyShell(save_path, dev_portal_browser)
    mysh.cmdloop()


if __name__ == "__main__":
    main()
